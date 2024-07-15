"""
This script is a modification of work originally created by Bill Peebles, Saining Xie, and Ikko Eltociear Ashimine

Original work licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

Original source: https://github.com/facebookresearch/DiT
License: https://creativecommons.org/licenses/by-nc/4.0/
"""

import os
import argparse

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models.edimt import EDiMT_models


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = EDiMT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    
    # Load a E-DiMT checkpoint:
    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    if "ema" in state_dict: # supports checkpoints from train.py
        state_dict = state_dict["ema"]
    model.load_state_dict(state_dict)
    model.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save images
    samples_path = args.ckpt.replace("checkpoints", "samples")
    samples_path = os.path.splitext(samples_path)[0] + ".png"
    samples_dir = os.path.dirname(samples_path)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    save_image(samples, samples_path, nrow=4, normalize=True, value_range=(-1, 1))


# To sample from the EMA weights of a custom 256x256 EDiM-L/2 model, run:
# python sample.py --model EDiMT-L/2 --image-size 256 --ckpt /path/to/model.pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(EDiMT_models.keys()), default="EDiMT-L/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
