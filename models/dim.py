"""
This script is a modification of work originally created by Bill Peebles, Saining Xie, and Ikko Eltociear Ashimine

Original work licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

Original source: https://github.com/facebookresearch/DiT
License: https://creativecommons.org/licenses/by-nc/4.0/

References:
GLIDE: https://github.com/openai/glide-text2im
MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""

import math
from functools import partial

import torch
import torch.nn as nn

import numpy as np

from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.layers import DropPath

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.modules.mamba_simple import Mamba

from layers import modulate, TimestepEmbedder, LabelEmbedder



#################################################################################
#                                 Core DiM Model                                #
#################################################################################
class DiMBlock(nn.Module):
    def __init__(self, dim, mixer_cls, d_state, d_conv, expand, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0., mlp_ratio=4.):

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = norm_cls(dim)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, hidden_states, cond, residual=None, inference_params=None):
        """Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
            cond: conditional embeddings
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)

        hidden_states = gate_msa.unsqueeze(1) * self.mixer(modulate(hidden_states, shift_msa, scale_msa), inference_params=inference_params)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(hidden_states), shift_mlp, scale_mlp))

        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        d_state=16,
        d_conv=4,
        expand=2,
        residual_in_fp32=False,
        fused_add_norm=False,
        mlp_ratio=4.,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="v1",
        if_devide_out=False,
        init_layer_scale=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
                        init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = DiMBlock(
        d_model,
        mixer_cls,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        mlp_ratio=mlp_ratio,
    )
    block.layer_idx = layer_idx
    return block


class FinalLayerDiM(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class DiM(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 patch_size,
                 hidden_size,
                 depth,
                 in_channels=4,
                 mlp_ratio=4.0,
                 class_dropout_prob=0.1,
                 learn_sigma=True,
                 #-- mamba init
                 ssm_cfg=None,
                 drop_path_rate=0.1,
                 norm_epsilon:float=1e-5,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 if_bidirectional=False,
                 bimamba_type="v1",
                 if_devide_out=False,
                 init_layer_scale=None,
                 rms_norm=True,
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 device=None,
                 dtype=None,
                 **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional

        self.d_model = self.num_features = self.hidden_size = hidden_size  # num_features for consistency with other models
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # Mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    hidden_size,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    residual_in_fp32=residual_in_fp32,
                    mlp_ratio=mlp_ratio,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            hidden_size, eps=norm_epsilon, **factory_kwargs
        )
        self.final_layer = FinalLayerDiM(hidden_size, patch_size, self.out_channels, norm_epsilon)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiM blocks:
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)

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

    def forward(self, x, t, y, inference_params=None):
        """
        Forward pass of DiM.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)                 # (N, T, D), where T = H * W / patch_size ** 2 and D is the hidden_size
        B, M, _ = x.shape

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:
                hidden_states, residual = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(layer), hidden_states, c, residual, inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                hidden_states_f, residual_f = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(self.layers[i * 2]), hidden_states, c, residual, inference_params
                )

                hidden_states_b, residual_b = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(self.layers[i * 2 + 1]), hidden_states.flip([1]), c, None if residual is None else residual.flip([1]), inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        hidden_states = self.final_layer(hidden_states, c)    # (N, T, patch_size ** 2 * out_channels)
        hidden_states = self.unpatchify(hidden_states)        # (N, out_channels, H, W)
        return hidden_states

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiM, but also batches the unconditional forward pass for classifier-free guidance.
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
#                                   DiM Configs                                  #
#################################################################################
def DiM_XL_2(**kwargs):
    return DiM(patch_size=2, hidden_size=1152, depth=28, **kwargs)

def DiM_L_2(**kwargs):
    return DiM(patch_size=2, hidden_size=960, depth=24, **kwargs)

def DiM_B_2(**kwargs):
    return DiM(patch_size=2, hidden_size=768, depth=12, **kwargs)

def DiM_S_2(**kwargs):
    return DiM(patch_size=2, hidden_size=324, depth=12, **kwargs)


DiM_models = {
    'DiM-XL/2': DiM_XL_2,
    'DiM-L/2':  DiM_L_2,
    'DiM-B/2':  DiM_B_2,
    'DiM-S/2':  DiM_S_2,
}
