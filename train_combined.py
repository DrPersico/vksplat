"""Train VkSplat on combined living room dataset (photos + video frames)."""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vksplat"))

from simple_trainer import TrainerConfig, MCMCTrainerConfig, train, join_dir, PRINT
from datetime import datetime

def train_at_resolution(tag="combined", steps=30000,
                        strategy="default", cap_max=2000000, skip_eval=False,
                        grow_grad2d=0.0002):
    if strategy == "mcmc":
        config = MCMCTrainerConfig()
        config.cap_max = cap_max
    else:
        config = TrainerConfig()

    config.dataset_dir = r"E:\vksplat_data\livingroom_combined"
    config.image_dir = ""
    config.sparse_dir = "sparse/1_clean"
    config.output_dir = rf"E:\vksplat_output\livingroom_{tag}"
    config.train_steps = steps
    config.save_train_renders = False
    config.ssim_lambda = 0.2
    config.grow_grad2d = grow_grad2d

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
    p.add_argument("--tag", default="combined")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--strategy", default="default", choices=["default", "mcmc"])
    p.add_argument("--cap-max", type=int, default=2000000)
    p.add_argument("--grow-grad2d", type=float, default=0.0002)
    p.add_argument("--skip-eval", action="store_true", help="Skip LPIPS/SSIM eval to save RAM")
    args = p.parse_args()
    train_at_resolution(args.tag, args.steps, args.strategy,
                        args.cap_max, args.skip_eval, args.grow_grad2d)
