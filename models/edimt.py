"""
This script is a modification of work originally created by Bill Peebles, Saining Xie, and Ikko Eltociear Ashimine

Original work licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

Original source: https://github.com/facebookresearch/DiT
License: https://creativecommons.org/licenses/by-nc/4.0/

References:
GLIDE: https://github.com/openai/glide-text2im
MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""

from functools import partial
import math

import torch
import torch.nn as nn

import numpy as np

from timm.models.vision_transformer import PatchEmbed, Attention
from timm.models.layers import DropPath

from mamba_ssm.ops.triton.layernorm import RMSNorm
from mamba_ssm.modules.mamba_simple import Mamba

from layers import modulate, modulate_2d, TimestepEmbedder, LabelEmbedder, get_2d_sincos_pos_embed, FinalLayer



#################################################################################
#                                Core EDiMT Model                               #
#################################################################################
class GMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(-1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class GatedCNNBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        x = x + shortcut
        x = x.permute(0, 3, 1, 2)
        return x


class SFEBlock(nn.Module):
    """
    A shallow feature extraction block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, input_size, **block_kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=[hidden_size, input_size, input_size], elementwise_affine=False, eps=1e-6)
        self.gcnn = GatedCNNBlock(dim=hidden_size)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
    def forward(self, x, c):
        shift_gcnn, scale_gcnn, gate_gcnn = self.adaLN_modulation(c).chunk(3, dim=1)
        x = x + gate_gcnn.unsqueeze(2).unsqueeze(2) * self.gcnn(modulate_2d(self.norm(x), shift_gcnn, scale_gcnn))
        return x
     
    
class EDiMTBlock(nn.Module):
    """
    A EDiMT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.norm3 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.ssm = Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2, bimamba_type="v2")
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.gmlp = GMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_ssm, scale_ssm, gate_ssm, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_ssm.unsqueeze(1) * self.ssm(modulate(self.norm2(x), shift_ssm, scale_ssm))
        x = x + gate_mlp.unsqueeze(1) * self.gmlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class EDiMT(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        patch_size,
        hidden_size,
        num_heads,
        depth,
        sfe_depth,
        in_channels=4,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, hidden_size, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=1, stride=1, padding=0)
        self.sfeblocks = nn.ModuleList([
            SFEBlock(hidden_size=hidden_size, input_size=input_size) for _ in range(sfe_depth)
        ])
                
        self.blocks = nn.ModuleList([
            EDiMTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer_dit(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers in SFE blocks:
        for block in self.sfeblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out adaLN modulation layers in EDiMT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of EDiMT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        
        x = self.proj(x)
        for block in self.sfeblocks:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)   # (N, D, H, W)
            
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)       # (N, T, D)

        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of E-DiMT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



#################################################################################
#                                 EDiMT Configs                                 #
#################################################################################
def EDiMT_XL_2(**kwargs):
    return EDiMT(patch_size=2, hidden_size=1152, num_heads=16, depth=28, sfe_depth=8, **kwargs)

def EDiMT_L_2(**kwargs):
    return EDiMT(patch_size=2, hidden_size=1024, num_heads=16, depth=11, sfe_depth=8, **kwargs)

def EDiMT_B_2(**kwargs):
    return EDiMT(patch_size=2, hidden_size=768, num_heads=12, depth=12, sfe_depth=8, **kwargs)

def EDiMT_S_2(**kwargs):
    return EDiMT(patch_size=2, hidden_size=384, num_heads=6, depth=12, sfe_depth=8, **kwargs)


EDiMT_models = {
    'EDiMT-XL/2': EDiMT_XL_2,
    'EDiMT-L/2':  EDiMT_L_2,
    'EDiMT-B/2':  EDiMT_B_2,
    'EDiMT-S/2':  EDiMT_S_2,
}


