"""Train VkSplat on video frames dataset at various resolutions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vksplat"))

from simple_trainer import TrainerConfig, MCMCTrainerConfig, train_main, train, eval, join_dir, TRAIN_DEVICE
from dataclasses import asdict
import json

def train_default(resolution_tag="4k", image_dir="images", skip_eval=False):
    config = TrainerConfig()
    config.dataset_dir = r"E:\vksplat_data\frames"
    config.image_dir = image_dir
    config.sparse_dir = "sparse/0"
    config.output_dir = rf"E:\vksplat_output\frames_{resolution_tag}"
    config.train_steps = 30000
    config.save_train_renders = False
    config.ssim_lambda = 0.2
    if skip_eval:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = join_dir(config.output_dir, f"{stamp}_frames")
        config.image_dir = join_dir(config.dataset_dir, config.image_dir)
        config.mask_dir = join_dir(config.dataset_dir, config.mask_dir) if config.mask_dir is not None else ""
        config.sparse_dir = join_dir(config.dataset_dir, config.sparse_dir)
        config.output_ply = os.path.join(config.output_dir, config.output_ply)
        os.makedirs(config.output_dir, exist_ok=True)
        with open(os.path.join(config.output_dir, 'config.json'), 'w') as fp:
            json.dump(asdict(config), fp, indent=4)
        train(config)
        print("Skipping eval to avoid OOM at 8K resolution.")
    else:
        train_main(config)

def train_mcmc(resolution_tag="4k", image_dir="images", cap_max=3000000):
    config = MCMCTrainerConfig()
    config.dataset_dir = r"E:\vksplat_data\frames"
    config.image_dir = image_dir
    config.sparse_dir = "sparse/0"
    config.output_dir = rf"E:\vksplat_output\frames_mcmc_{resolution_tag}"
    config.train_steps = 30000
    config.cap_max = cap_max
    config.save_train_renders = False
    train_main(config)

if __name__ == "__main__":
    # Phase 3: train at 8K, skip eval (LPIPS at 8K crashes)
    train_default(resolution_tag="8k", image_dir="images", skip_eval=True)
