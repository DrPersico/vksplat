"""Systematic Brush parameter sweep on the kitchen_v3 dataset.

Tests different configurations and computes eval metrics for each.
"""

import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

BRUSH_EXE = r"E:\brush\target\release\brush.exe"
DATASET = r"E:\vksplat_data\kitchen_v3"
OUTPUT_BASE = r"E:\vksplat_output\brush_sweep"

CONFIGS = {
    "res960_30k_1M": {
        "total_train_iters": 30000, "max_splats": 1000000,
        "ssim_weight": 0.2, "max_resolution": 960,
    },
    "res960_30k_3M": {
        "total_train_iters": 30000, "max_splats": 3000000,
        "ssim_weight": 0.2, "max_resolution": 960,
    },
    "res960_50k_3M": {
        "total_train_iters": 50000, "max_splats": 3000000,
        "ssim_weight": 0.2, "max_resolution": 960,
    },
    "res960_30k_3M_ssim05": {
        "total_train_iters": 30000, "max_splats": 3000000,
        "ssim_weight": 0.5, "max_resolution": 960,
    },
    "res960_50k_5M_sh3": {
        "total_train_iters": 50000, "max_splats": 5000000,
        "ssim_weight": 0.2, "max_resolution": 960, "sh_degree": 3,
    },
    "res1920_30k_3M": {
        "total_train_iters": 30000, "max_splats": 3000000,
        "ssim_weight": 0.2, "max_resolution": 1920,
    },
    "res1920_50k_5M": {
        "total_train_iters": 50000, "max_splats": 5000000,
        "ssim_weight": 0.2, "max_resolution": 1920,
    },
    "res1920_30k_3M_ssim05": {
        "total_train_iters": 30000, "max_splats": 3000000,
        "ssim_weight": 0.5, "max_resolution": 1920,
    },
}


def compute_metrics(eval_dir: str, gt_image_dir: str) -> dict:
    """Compute PSNR and SSIM from Brush eval renders vs ground truth."""
    from skimage.metrics import structural_similarity

    renders = sorted([f for f in os.listdir(eval_dir)
                      if f.lower().endswith(('.png', '.jpg'))])
    if not renders:
        return {}

    psnrs, ssims = [], []
    for render_name in renders:
        render = cv2.imread(os.path.join(eval_dir, render_name))
        if render is None:
            continue

        gt_name = render_name[:-4] if render_name.endswith(".png") and ".jpg.png" in render_name else render_name
        gt_path = os.path.join(gt_image_dir, gt_name)
        if not os.path.exists(gt_path):
            continue

        gt = cv2.imread(gt_path)
        if gt is None:
            continue
        if render.shape != gt.shape:
            gt = cv2.resize(gt, (render.shape[1], render.shape[0]),
                            interpolation=cv2.INTER_AREA)

        mse = np.mean((render.astype(float) - gt.astype(float)) ** 2)
        if mse > 0:
            psnrs.append(10 * np.log10(255.0 ** 2 / mse))
        ssims.append(structural_similarity(render, gt, channel_axis=2, data_range=255))

    if not psnrs:
        return {}
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
            "n_eval": len(psnrs)}


def run_config(name: str, cfg: dict) -> dict:
    export_dir = os.path.join(OUTPUT_BASE, name)
    os.makedirs(export_dir, exist_ok=True)

    cmd = [
        BRUSH_EXE, DATASET,
        "--total-train-iters", str(cfg["total_train_iters"]),
        "--max-splats", str(cfg["max_splats"]),
        "--ssim-weight", str(cfg["ssim_weight"]),
        "--max-resolution", str(cfg["max_resolution"]),
        "--export-every", str(cfg["total_train_iters"]),
        "--export-path", export_dir,
        "--eval-split-every", "8",
        "--eval-every", str(cfg["total_train_iters"]),
        "--eval-save-to-disk",
        "--seed", "42",
    ]
    if "sh_degree" in cfg:
        cmd.extend(["--sh-degree", str(cfg["sh_degree"])])

    print(f"\n{'='*60}")
    print(f"CONFIG: {name}")
    print(f"  iters={cfg['total_train_iters']}, splats={cfg['max_splats']}, "
          f"ssim={cfg['ssim_weight']}, res={cfg['max_resolution']}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    print(f"  Finished in {elapsed:.0f}s (rc={result.returncode})")

    metrics = {"elapsed": elapsed, "returncode": result.returncode, **cfg}

    final_eval = os.path.join(export_dir, f"eval_{cfg['total_train_iters']}")
    if os.path.isdir(final_eval):
        res = cfg["max_resolution"]
        gt_dir = os.path.join(DATASET, "images_4" if res <= 960 else "images_2")
        m = compute_metrics(final_eval, gt_dir)
        if m:
            metrics.update(m)
            print(f"  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.3f}  ({m['n_eval']} images)")

    plys = [f for f in os.listdir(export_dir) if f.endswith(".ply")]
    if plys:
        ply_path = os.path.join(export_dir, sorted(plys)[-1])
        metrics["ply_size_mb"] = os.path.getsize(ply_path) / 1e6

    return metrics


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                   help="Which configs to run")
    args = p.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("=" * 60)
    print("BRUSH PARAMETER SWEEP")
    print(f"Dataset: {DATASET}")
    print(f"Configs: {len(args.configs)}")
    print("=" * 60)

    all_results = {}
    for name in args.configs:
        if name not in CONFIGS:
            print(f"Unknown config: {name}")
            continue
        all_results[name] = run_config(name, CONFIGS[name])

    # Summary table
    print(f"\n{'='*60}")
    print("SWEEP RESULTS")
    print(f"{'='*60}")
    header = f"{'Config':<28} {'Res':>4} {'Steps':>6} {'Splats':>6} {'SSIM_w':>6} {'PSNR':>7} {'SSIM':>7} {'Time':>7}"
    print(header)
    print("-" * len(header))

    for name, r in sorted(all_results.items(), key=lambda x: -x[1].get("psnr", 0)):
        res = r.get("max_resolution", 0)
        steps = r.get("total_train_iters", 0)
        splats = r.get("max_splats", 0)
        ssim_w = r.get("ssim_weight", 0)
        psnr = r.get("psnr", 0)
        ssim = r.get("ssim", 0)
        elapsed = r.get("elapsed", 0)
        splats_s = f"{splats/1e6:.0f}M"
        psnr_s = f"{psnr:.2f}" if psnr else "n/a"
        ssim_s = f"{ssim:.3f}" if ssim else "n/a"
        print(f"{name:<28} {res:>4} {steps:>6} {splats_s:>6} {ssim_w:>6.1f} "
              f"{psnr_s:>7} {ssim_s:>7} {elapsed:>6.0f}s")

    results_path = os.path.join(OUTPUT_BASE, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
