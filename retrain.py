"""
Iterative training: load a trained PLY and continue training with fresh LR + refinement.

This is NOT the same as simply adding more steps to the original training:
- Fresh learning rate schedule (means can move aggressively again)
- Fresh refinement window (densification can split/clone/prune to fix blurry regions)
- Fresh Adam optimizer state (escape local minima)

Usage:
    python retrain.py --ply path/to/splat.ply --dataset-dir path/to/dataset [options]

The script loads the trained PLY, injects its splats into a fresh training session
(using the same dataset for images/cameras), and trains for additional steps.

RECOMMENDED CONFIGURATIONS:

1. Fine-tune a converged model (preserve quality, minor improvements):
   python retrain.py --ply splat.ply --dataset-dir data/ --steps 5000 --strategy mcmc
       --noise-lr 0 --refine-start 99999 --refine-stop 99999

2. Fix blur by adding capacity (default strategy densification):
   python retrain.py --ply splat.ply --dataset-dir data/ --steps 10000 --strategy default
       --grow-grad2d 0.0004 --means-lr 3e-5

3. Retrain an UNDER-CONVERGED model (stopped early, has clear blur):
   python retrain.py --ply splat.ply --dataset-dir data/ --steps 15000 --strategy default
       --means-lr 8e-5 --step-offset 0

IMPORTANT NOTES:
- The default means_lr is 10x lower than fresh training (1.6e-5 vs 1.6e-4)
  because fresh Adam optimizer state destabilizes converged models at high LR.
- MCMC noise_lr defaults to 0 (disabled) to prevent noise from destroying the model.
- Opacity reset is disabled (stop_reset_at=0) to prevent clamping converged opacities.
- For fully converged models at capacity, consider training fresh with higher cap_max
  instead of iterative retraining at the same capacity.
"""

import sys
import os
import argparse
import json
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vksplat"))

from simple_trainer import TrainerConfig, MCMCTrainerConfig, join_dir, PRINT
from dataclasses import asdict
from datetime import datetime
from time import perf_counter
from tqdm import tqdm
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

# Add parent dir for ply_utils
sys.path.insert(0, os.path.dirname(__file__))
from ply_utils import read_ply, print_ply_stats


TRAIN_DEVICE = -1


def retrain(config: TrainerConfig, ply_path: str, step_offset: int = 0):
    """
    Load a trained PLY and continue training with fresh optimizer state.
    
    Args:
        config: Training configuration (dataset paths, hyperparameters)
        ply_path: Path to the trained splat.ply to use as initialization
        step_offset: Step number to start at (affects SH degree ramp and refinement timing).
                     Use 3000+ to skip the SH degree ramp (all degrees active immediately).
    """
    import vksplat

    # Load PLY
    PRINT(f"Loading PLY: {ply_path}")
    ply_data = read_ply(ply_path)
    print_ply_stats(ply_data)
    PRINT()

    # Create and initialize module
    module = vksplat.VkSplat()
    spv_dir = join_dir(os.path.join(os.path.dirname(__file__), "vksplat"), 'shader')
    module.initialize(spv_dir, TRAIN_DEVICE)

    # Load training data (this also initializes splats from COLMAP sparse, which we'll override)
    train_meta = module.set_train_config(asdict(config))
    for key, value in train_meta.items():
        if isinstance(value, np.ndarray):
            train_meta[key] = value.tolist()
    PRINT(f"Dataset loaded: {module.num_train} train, {module.num_val} val images")
    PRINT()

    # Clean NaN/inf splats before injection
    valid_mask = np.isfinite(ply_data['xyz']).all(axis=1)
    valid_mask &= np.isfinite(ply_data['scales']).all(axis=1)
    valid_mask &= np.isfinite(ply_data['opacity'])
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        PRINT(f"Removing {n_invalid} invalid (NaN/inf) splats before injection")
        ply_data = {
            'xyz': ply_data['xyz'][valid_mask],
            'sh_coeffs': ply_data['sh_coeffs'][valid_mask],
            'opacity': ply_data['opacity'][valid_mask],
            'scales': ply_data['scales'][valid_mask],
            'rotations': ply_data['rotations'][valid_mask],
            'num_splats': int(valid_mask.sum()),
        }

    # Override splats with PLY data
    PRINT(f"Injecting {ply_data['num_splats']} splats from PLY (replacing {len(module.opacities)} from sparse init)")
    module.set_gauss_params(
        ply_data['xyz'].astype(np.float32),
        ply_data['rotations'].astype(np.float32),
        ply_data['scales'].astype(np.float32),
        ply_data['opacity'].reshape(-1, 1).astype(np.float32),
        ply_data['sh_coeffs'].astype(np.float32),
    )
    PRINT(f"Splats injected. Current count: {len(module.opacities)}")
    PRINT()

    # Save config
    os.makedirs(config.output_dir, exist_ok=True)
    retrain_config = asdict(config)
    retrain_config['_retrain_from'] = ply_path
    retrain_config['_step_offset'] = step_offset
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as fp:
        json.dump(retrain_config, fp, indent=4)
    with open(os.path.join(config.output_dir, 'train.json'), 'w') as fp:
        json.dump(train_meta, fp, indent=4)

    # Train
    PRINT(f"Starting retrain: {config.train_steps} steps (step_offset={step_offset})")
    PRINT(f"  LR schedule: max_steps={config.max_steps}, means_lr={config.means_lr} -> {config.means_lr_final}")
    PRINT(f"  Refinement: start={config.refine_start_iter}, stop={config.refine_stop_iter}, every={config.refine_every}")
    PRINT()

    t0 = perf_counter()
    shuffle_idx = list(range(module.num_train))
    for i in tqdm(range(config.train_steps), "Retraining"):
        if i > 0 and i % len(shuffle_idx) == 0:
            random.shuffle(shuffle_idx)
        image_idx = shuffle_idx[i % len(shuffle_idx)]
        step = i + step_offset
        module.train_step(image_idx, step)
    t1 = perf_counter()

    num_splats = len(module.opacities)
    vram_usage = module.get_vram_usage()
    peak_vram_usage = module.get_peak_vram_usage()
    PRINT(f"Time elapsed: {t1-t0:.1f} seconds")
    PRINT(f"VRAM usage: {vram_usage/1024**3:.2f} GiB")
    PRINT(f"Num splats: {num_splats}")

    with open(os.path.join(config.output_dir, 'train.json'), 'w') as fp:
        train_json = {
            'num_splats': num_splats,
            'time_elapsed': t1-t0,
            'vram': vram_usage,
            'peak_vram': peak_vram_usage,
            'breakdown': module.get_timing_breakdown(),
            'vram_breakdown': module.get_vram_breakdown(),
            "train_images": [module.get_train_image_path(i) for i in range(module.num_train)],
            "val_images": [module.get_val_image_path(i) for i in range(module.num_val)],
        }
        train_json.update(train_meta)
        json.dump(train_json, fp, indent=4)
    PRINT()

    # Write output PLY
    PRINT("Writing PLY")
    module.write_ply(config.output_ply)
    PRINT()

    # Save validation renders
    def save_one_image(pixel_data, path):
        import cv2
        im = pixel_data.copy()
        im[:, :, 3] = 1.0 - im[:, :, 3]
        im = np.round(65535 * np.clip(im, 0, 1)).astype(np.uint16)
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
        cv2.imwrite(path, im)

    rendered_images = []
    for i in tqdm(range(module.num_val), "Rendering val images"):
        module.render_val(i)
        path = os.path.join(config.output_dir, f'val_{i:05d}.png')
        rendered_images.append((path, module.pixel_state))

    with ThreadPoolExecutor() as executor:
        futures = []
        for path, pixel_data in rendered_images:
            futures.append(executor.submit(save_one_image, pixel_data, path))
        for future in tqdm(futures, "Saving val images"):
            future.result()
    PRINT()

    module.cleanup()
    PRINT("Done.")


def main():
    p = argparse.ArgumentParser(
        description="Iterative training: refine a trained model with fresh LR + densification")
    p.add_argument("--ply", required=True, help="Path to trained splat.ply")
    p.add_argument("--dataset-dir", required=True, help="Dataset root (images + sparse)")
    p.add_argument("--image-dir", default="images_2", help="Subdirectory with training images")
    p.add_argument("--sparse-dir", default=None,
                   help="Sparse reconstruction dir (default: auto from image-dir)")
    p.add_argument("--output-base", default=r"E:\vksplat_output",
                   help="Parent output directory")
    p.add_argument("--tag", default="retrain", help="Tag for output folder")

    # Training params
    p.add_argument("--steps", type=int, default=10000,
                   help="Number of additional training steps (default: 10000)")
    p.add_argument("--step-offset", type=int, default=0,
                   help="Starting step counter. Default 0 re-ramps SH degrees "
                        "(0->3 over first 3000 steps) which provides natural warmup "
                        "for fresh Adam state. Set to 3000+ to skip the ramp.")
    p.add_argument("--strategy", default="default", choices=["default", "mcmc"])
    p.add_argument("--cap-max", type=int, default=2000000, help="MCMC cap_max")

    # LR tuning - defaults are 10x LOWER than fresh training since we're fine-tuning
    # a converged model. Fresh Adam state + high LR = catastrophic degradation.
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max_steps for LR schedule (default: step-offset + steps)")
    p.add_argument("--means-lr", type=float, default=1.6e-5,
                   help="Initial means LR (default 10x lower than fresh training "
                        "to prevent destabilizing a converged model)")
    p.add_argument("--means-lr-final", type=float, default=1.6e-7,
                   help="Final means LR")
    p.add_argument("--ssim-lambda", type=float, default=0.2)

    # MCMC noise (set to 0 to disable noise injection for short retrains)
    p.add_argument("--noise-lr", type=float, default=0.0,
                   help="MCMC noise injection LR. Default 0 (disabled) to prevent "
                        "noise from destabilizing short retrains. Original default is 5e5.")

    # Refinement tuning
    p.add_argument("--refine-start", type=int, default=100,
                   help="Step (relative to step_offset) to start refinement")
    p.add_argument("--refine-stop", type=int, default=None,
                   help="Step (relative to step_offset) to stop refinement. "
                        "Default: 70%% of total steps + step_offset")
    p.add_argument("--refine-every", type=int, default=100)
    p.add_argument("--grow-grad2d", type=float, default=0.0002,
                   help="Densification gradient threshold. Lower = more splitting")
    p.add_argument("--prune-opa", type=float, default=0.005,
                   help="Opacity below which splats get pruned")
    p.add_argument("--reset-every", type=int, default=3000,
                   help="Opacity reset interval")

    # Eval
    p.add_argument("--skip-eval", action="store_true")

    args = p.parse_args()

    # Build config
    if args.strategy == "mcmc":
        config = MCMCTrainerConfig()
        config.cap_max = args.cap_max
    else:
        config = TrainerConfig()

    config.dataset_dir = args.dataset_dir
    config.image_dir = args.image_dir
    if args.sparse_dir:
        config.sparse_dir = args.sparse_dir
    else:
        sparse_map = {
            "images": "sparse/0_fullres",
            "images_subset": "sparse/0_fullres_subset",
        }
        config.sparse_dir = sparse_map.get(args.image_dir, "sparse/0")

    config.train_steps = args.steps
    config.max_steps = args.max_steps if args.max_steps else (args.step_offset + args.steps)
    config.means_lr = args.means_lr
    config.means_lr_final = args.means_lr_final
    config.ssim_lambda = args.ssim_lambda
    config.noise_lr = args.noise_lr

    # Refinement timing (absolute step values, accounting for offset)
    # refine_start should be shortly after step_offset so densification begins quickly
    config.refine_start_iter = args.step_offset + args.refine_start
    if args.refine_stop is not None:
        config.refine_stop_iter = args.step_offset + args.refine_stop
    else:
        # Stop refinement at 70% of total training window
        config.refine_stop_iter = args.step_offset + int(args.steps * 0.7)
    # Sanity check: stop must be > start
    if config.refine_stop_iter <= config.refine_start_iter:
        config.refine_stop_iter = config.refine_start_iter + max(args.steps // 2, 1000)
    config.refine_every = args.refine_every
    config.grow_grad2d = args.grow_grad2d
    config.prune_opa = args.prune_opa
    config.reset_every = args.reset_every
    # Disable opacity reset for retrain - it clamps all opacities to 2*prune_opa
    # which destroys the converged model. The original training uses it to allow
    # gaussians to be re-evaluated, but for retrain the model is already converged.
    config.stop_reset_at = 0

    # Output dir
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = join_dir(args.output_base, f"{dataset_name}_{args.tag}")
    config.output_dir = join_dir(config.output_dir, f"{stamp}_{dataset_name}")
    PRINT("Work dir:", config.output_dir)
    PRINT()

    # Resolve paths
    config.image_dir = join_dir(config.dataset_dir, config.image_dir)
    config.mask_dir = join_dir(config.dataset_dir, config.mask_dir) if config.mask_dir else ""
    config.sparse_dir = join_dir(config.dataset_dir, config.sparse_dir)
    config.output_ply = os.path.join(config.output_dir, "splat.ply")

    retrain(config, args.ply, step_offset=args.step_offset)

    if not args.skip_eval:
        PRINT("Running eval...")
        import subprocess
        eval_script = os.path.join(os.path.dirname(__file__), "run_eval.py")
        subprocess.run([sys.executable, eval_script, config.output_dir], check=True)


if __name__ == "__main__":
    main()
