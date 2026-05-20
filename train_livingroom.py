"""Train VkSplat on living room photos dataset."""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vksplat"))

from simple_trainer import TrainerConfig, MCMCTrainerConfig, train, join_dir, PRINT
from datetime import datetime

def train_at_resolution(image_dir="images_2", tag="1080p", steps=30000,
                        strategy="default", cap_max=2000000, skip_eval=False,
                        grow_grad2d=0.0002, refine_stop_iter=None,
                        dataset_dir=r"E:\vksplat_data\livingroom",
                        output_base=r"E:\vksplat_output",
                        ssim_lambda=0.2, opacity_reg=None, scale_reg=None,
                        noise_lr=None, max_steps=None,
                        image_cache_device="cpu"):
    if strategy == "mcmc":
        config = MCMCTrainerConfig()
        config.cap_max = cap_max
    else:
        config = TrainerConfig()

    config.dataset_dir = dataset_dir
    config.image_dir = image_dir
    sparse_map = {
        "images": "sparse/0_fullres",
        "images_subset": "sparse/0_fullres_subset",
    }
    config.sparse_dir = sparse_map.get(image_dir, "sparse/0")
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    config.output_dir = os.path.join(output_base, f"{dataset_name}_{tag}")
    config.train_steps = steps
    if max_steps is not None:
        config.max_steps = max_steps
    config.save_train_renders = False
    config.ssim_lambda = ssim_lambda
    config.grow_grad2d = grow_grad2d
    if refine_stop_iter is not None:
        config.refine_stop_iter = refine_stop_iter
    # MCMC anti-fog regularizers / relocation noise (only meaningful for mcmc)
    if opacity_reg is not None:
        config.opacity_reg = opacity_reg
    if scale_reg is not None:
        config.scale_reg = scale_reg
    if noise_lr is not None:
        config.noise_lr = noise_lr
    config.image_cache_device = image_cache_device

    dataset_name = os.path.basename(os.path.normpath(config.dataset_dir))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = join_dir(config.output_dir, f"{stamp}_{dataset_name}")
    PRINT("Work dir:", config.output_dir)
    PRINT()

    config.image_dir = join_dir(config.dataset_dir, config.image_dir)
    config.mask_dir = join_dir(config.dataset_dir, config.mask_dir) if config.mask_dir is not None else ""
    config.sparse_dir = join_dir(config.dataset_dir, config.sparse_dir)
    config.output_ply = os.path.join(config.output_dir, config.output_ply)

    train(config)
    if not skip_eval:
        PRINT("Running eval in a subprocess to free training memory first...")
        import subprocess
        eval_script = os.path.join(os.path.dirname(__file__), "run_eval.py")
        subprocess.run([sys.executable, eval_script, config.output_dir], check=True)
    else:
        PRINT("Skipping eval (--skip-eval)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="images_2")
    p.add_argument("--tag", default="1080p")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--max-steps", type=int, default=None,
                   help="LR-decay schedule denominator. Default leaves it at "
                        "the config default (30000), so the decay schedule "
                        "clamps there and steps beyond run at the final "
                        "(slowest) LR. Set this = --steps to stretch the LR "
                        "decay over the full run instead.")
    p.add_argument("--strategy", default="default", choices=["default", "mcmc"])
    p.add_argument("--cap-max", type=int, default=2000000)
    p.add_argument("--grow-grad2d", type=float, default=0.0002)
    p.add_argument("--refine-stop-iter", type=int, default=None)
    p.add_argument("--skip-eval", action="store_true", help="Skip LPIPS/SSIM eval to save RAM")
    p.add_argument("--dataset-dir", default=r"E:\vksplat_data\livingroom",
                   help="Dataset root (must contain image-dir and sparse_dir subfolders)")
    p.add_argument("--output-base", default=r"E:\vksplat_output",
                   help="Parent directory for training outputs")
    p.add_argument("--ssim-lambda", type=float, default=0.2,
                   help="SSIM loss weight. Raise (0.4-0.5) for better "
                        "perceptual/LPIPS quality (may lower PSNR)")
    p.add_argument("--opacity-reg", type=float, default=None,
                   help="MCMC opacity regularizer (default 0.01). Raise "
                        "(~0.05) to fight degenerate fog at higher res")
    p.add_argument("--scale-reg", type=float, default=None,
                   help="MCMC scale regularizer (default 0.01)")
    p.add_argument("--noise-lr", type=float, default=None,
                   help="MCMC relocation noise lr (default 5e5). Lower "
                        "(~1e5) to reduce relocation churn / fog")
    p.add_argument("--image-cache-device", default="cpu",
                   choices=["cpu", "gpu"],
                   help="'gpu' pre-uploads all images to VRAM and frees CPU "
                        "copies (~9 GB VRAM for 3852 images_4 frames). "
                        "Eliminates per-step PCIe upload and sustained RAM. "
                        "Only use if dataset fits in VRAM alongside splats.")
    args = p.parse_args()
    train_at_resolution(args.image_dir, args.tag, args.steps, args.strategy,
                        args.cap_max, args.skip_eval, args.grow_grad2d,
                        args.refine_stop_iter, args.dataset_dir, args.output_base,
                        args.ssim_lambda, args.opacity_reg, args.scale_reg,
                        args.noise_lr, args.max_steps,
                        args.image_cache_device)
