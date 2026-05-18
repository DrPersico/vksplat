"""Train VkSplat on phone photos dataset at various resolutions."""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vksplat"))

from simple_trainer import TrainerConfig, MCMCTrainerConfig, train, eval, join_dir, PRINT
from datetime import datetime

def train_at_resolution(image_dir="images_4", tag="1080p", steps=30000,
                        strategy="default", cap_max=3000000, skip_eval=False):
    if strategy == "mcmc":
        config = MCMCTrainerConfig()
        config.cap_max = cap_max
    else:
        config = TrainerConfig()

    config.dataset_dir = r"E:\vksplat_data\photos"
    config.image_dir = image_dir
    config.sparse_dir = "sparse/0"
    config.output_dir = rf"E:\vksplat_output\photos_{tag}"
    config.train_steps = steps
    config.save_train_renders = False
    config.ssim_lambda = 0.2

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
        import subprocess, json
        eval_script = os.path.join(os.path.dirname(__file__), "run_eval.py")
        subprocess.run([sys.executable, eval_script, config.output_dir], check=True)
    else:
        PRINT("Skipping eval (--skip-eval)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="images_4")
    p.add_argument("--tag", default="1080p")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--strategy", default="default", choices=["default", "mcmc"])
    p.add_argument("--cap-max", type=int, default=3000000)
    p.add_argument("--skip-eval", action="store_true", help="Skip LPIPS/SSIM eval to save RAM")
    args = p.parse_args()
    train_at_resolution(args.image_dir, args.tag, args.steps, args.strategy, args.cap_max, args.skip_eval)
