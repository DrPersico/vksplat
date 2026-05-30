"""Compare vksplat vs Brush training on the same COLMAP dataset.

Runs both trainers with multiple configurations and collects metrics
for a fair quality assessment.

Usage:
  python run_brush_comparison.py E:\\vksplat_data\\kitchen_v3
  python run_brush_comparison.py E:\\vksplat_data\\kitchen_v3 --configs A B C
  python run_brush_comparison.py E:\\vksplat_data\\kitchen_v3 --brush-only
  python run_brush_comparison.py E:\\vksplat_data\\kitchen_v3 --vksplat-only
"""

import argparse
import json
import os
import subprocess
import sys
import time


BRUSH_EXE = r"E:\brush\target\release\brush.exe"

CONFIGS = {
    "A": {
        "description": "Matched baseline (30K steps, 1M splats, ssim=0.2)",
        "vksplat": {
            "steps": 30000, "cap_max": 1000000, "ssim_lambda": 0.2,
            "n_runs": 1, "image_dir": "images_4",
        },
        "brush": {
            "total_train_iters": 30000, "max_splats": 1000000,
            "ssim_weight": 0.2, "lpips_loss_weight": 0.0,
            "max_resolution": 960,
        },
    },
    "B": {
        "description": "Each trainer's sweet spot",
        "vksplat": {
            "steps": 100000, "cap_max": 2000000, "ssim_lambda": 0.4,
            "n_runs": 1, "image_dir": "images_4",
        },
        "brush": {
            "total_train_iters": 30000, "max_splats": 3000000,
            "ssim_weight": 0.5, "lpips_loss_weight": 0.0,
            "max_resolution": 960,
        },
    },
    "C": {
        "description": "Maximum quality",
        "vksplat": {
            "steps": 100000, "cap_max": 2000000, "ssim_lambda": 0.4,
            "n_runs": 3, "image_dir": "images_4",
        },
        "brush": {
            "total_train_iters": 50000, "max_splats": 5000000,
            "ssim_weight": 0.5, "lpips_loss_weight": 0.0,
            "max_resolution": 960, "sh_degree": 3,
        },
    },
}


def run_vksplat(dataset_dir: str, config_name: str, cfg: dict,
                output_base: str) -> dict:
    """Train vksplat with specified configuration."""
    train_script = os.path.join(os.path.dirname(__file__), "train_livingroom.py")
    tag = f"cmp_{config_name}_vksplat"

    cmd = [
        sys.executable, train_script,
        "--dataset-dir", dataset_dir,
        "--image-dir", cfg["image_dir"],
        "--strategy", "mcmc",
        "--cap-max", str(cfg["cap_max"]),
        "--ssim-lambda", str(cfg["ssim_lambda"]),
        "--steps", str(cfg["steps"]),
        "--max-steps", str(cfg["steps"]),
        "--image-cache-device", "gpu",
        "--tag", tag,
        "--output-base", output_base,
    ]

    print(f"\n  [vksplat config {config_name}] Starting training...")
    print(f"    cmd: {' '.join(cmd)}")
    sys.stdout.flush()

    t0 = time.time()
    n_runs = cfg.get("n_runs", 1)
    best_result = None

    for run_i in range(n_runs):
        if n_runs > 1:
            print(f"\n    Run {run_i+1}/{n_runs}...")
        result = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - t0

        tag_dir = os.path.join(output_base,
                               f"{os.path.basename(dataset_dir)}_{tag}")
        out_dir = None
        if os.path.isdir(tag_dir):
            subs = sorted(os.listdir(tag_dir))
            if subs:
                out_dir = os.path.join(tag_dir, subs[-1])

        if out_dir:
            eval_path = os.path.join(out_dir, "eval.json")
            if os.path.exists(eval_path):
                with open(eval_path) as f:
                    ev = json.load(f)
                mean = ev.get("mean", ev)
                lpips = float(mean.get("lpips_alex", mean.get("lpips", 99)))
                if best_result is None or lpips < best_result.get("lpips_alex", 99):
                    best_result = {
                        "output_dir": out_dir,
                        "elapsed": elapsed,
                        "returncode": result.returncode,
                        **{k: float(v) for k, v in mean.items()},
                    }

    return best_result or {"error": "no eval found", "elapsed": time.time() - t0}


def run_brush(dataset_dir: str, config_name: str, cfg: dict,
              output_base: str) -> dict:
    """Train Brush with specified configuration."""
    if not os.path.exists(BRUSH_EXE):
        return {"error": f"brush.exe not found at {BRUSH_EXE}"}

    export_dir = os.path.join(output_base, f"brush_cmp_{config_name}")
    os.makedirs(export_dir, exist_ok=True)

    cmd = [
        BRUSH_EXE,
        dataset_dir,
        "--total-train-iters", str(cfg["total_train_iters"]),
        "--max-splats", str(cfg["max_splats"]),
        "--ssim-weight", str(cfg["ssim_weight"]),
        "--max-resolution", str(cfg["max_resolution"]),
        "--export-every", "5000",
        "--export-path", export_dir,
        "--eval-split-every", "8",
        "--eval-every", "1000",
        "--eval-save-to-disk",
    ]
    if cfg.get("lpips_loss_weight", 0) > 0:
        cmd.extend(["--lpips-loss-weight", str(cfg["lpips_loss_weight"])])
    if "sh_degree" in cfg:
        cmd.extend(["--sh-degree", str(cfg["sh_degree"])])

    print(f"\n  [Brush config {config_name}] Starting training...")
    print(f"    cmd: {' '.join(cmd)}")
    sys.stdout.flush()

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    print(f"    Brush finished in {elapsed:.0f}s (rc={result.returncode})")

    metrics = {"elapsed": elapsed, "returncode": result.returncode,
               "export_dir": export_dir}

    # Find the final export PLY
    if os.path.isdir(export_dir):
        plys = sorted([f for f in os.listdir(export_dir) if f.endswith(".ply")])
        if plys:
            metrics["final_ply"] = os.path.join(export_dir, plys[-1])
            metrics["n_exports"] = len(plys)

    # Compute metrics from eval renders saved to disk
    eval_metrics = _compute_brush_eval_metrics(export_dir, dataset_dir, cfg)
    if eval_metrics:
        metrics.update(eval_metrics)
        print(f"    PSNR={eval_metrics.get('psnr',0):.2f} "
              f"SSIM={eval_metrics.get('ssim',0):.3f} "
              f"LPIPS={eval_metrics.get('lpips_alex',0):.3f}")

    if result.stderr:
        err_lines = result.stderr.strip().split("\n")
        if len(err_lines) <= 5:
            metrics["stderr"] = result.stderr.strip()
        else:
            metrics["stderr_tail"] = "\n".join(err_lines[-5:])

    return metrics


def _compute_brush_eval_metrics(export_dir: str, dataset_dir: str,
                                cfg: dict) -> dict:
    """Compute PSNR/SSIM/LPIPS from Brush's --eval-save-to-disk renders."""
    import cv2
    import numpy as np

    # Brush saves eval renders in eval_{iter}/ subdirectories; use the latest
    eval_render_dir = None
    if os.path.isdir(export_dir):
        eval_dirs = sorted([d for d in os.listdir(export_dir)
                            if d.startswith("eval_") and
                            os.path.isdir(os.path.join(export_dir, d))],
                           key=lambda d: int(d.split("_")[1]))
        if eval_dirs:
            eval_render_dir = os.path.join(export_dir, eval_dirs[-1])
            print(f"    Using eval renders from {eval_dirs[-1]}/")
    if not eval_render_dir:
        print(f"    No eval renders found in {export_dir}")
        return {}

    renders = sorted([f for f in os.listdir(eval_render_dir)
                      if f.lower().endswith(('.png', '.jpg'))])
    if not renders:
        print(f"    No eval render images found")
        return {}

    print(f"    Computing metrics from {len(renders)} eval renders...")

    # Brush names renders as "{gt_filename}.png" — match to GT by stem
    image_dir = os.path.join(dataset_dir, "images_4")
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(dataset_dir, "images_2")

    psnrs, ssims = [], []
    n_compared = 0
    for render_name in renders:
        render_path = os.path.join(eval_render_dir, render_name)
        render = cv2.imread(render_path)
        if render is None:
            continue

        # e.g. "v0_frame_00000.jpg.png" → GT is "v0_frame_00000.jpg"
        gt_name = render_name
        if gt_name.endswith(".png") and ".jpg.png" in gt_name:
            gt_name = gt_name[:-4]  # strip trailing .png
        gt_path = os.path.join(image_dir, gt_name)
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

        from skimage.metrics import structural_similarity
        ssim_val = structural_similarity(render, gt, channel_axis=2,
                                         data_range=255)
        ssims.append(ssim_val)
        n_compared += 1

    if not psnrs:
        print(f"    Could not compute metrics (no valid pairs)")
        return {}

    result = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "n_eval": n_compared,
    }
    print(f"    Computed from {n_compared} image pairs")
    return result


def main():
    p = argparse.ArgumentParser(description="vksplat vs Brush comparison")
    p.add_argument("dataset_dir", help="COLMAP dataset directory")
    p.add_argument("--configs", nargs="+", default=["A", "B", "C"],
                   choices=list(CONFIGS.keys()),
                   help="Which configs to test (default: A B C)")
    p.add_argument("--output-base", default=r"E:\vksplat_output",
                   help="Output directory for training results")
    p.add_argument("--brush-only", action="store_true",
                   help="Only run Brush (skip vksplat)")
    p.add_argument("--vksplat-only", action="store_true",
                   help="Only run vksplat (skip Brush)")
    args = p.parse_args()

    print("=" * 60)
    print("VKSPLAT vs BRUSH COMPARISON")
    print("=" * 60)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Configs: {', '.join(args.configs)}")
    print(f"Output:  {args.output_base}")

    # Verify dataset exists
    sparse_path = os.path.join(args.dataset_dir, "sparse", "0")
    images_path = os.path.join(args.dataset_dir, "images_2")
    if not os.path.exists(sparse_path):
        print(f"ERROR: {sparse_path} not found")
        sys.exit(1)
    if not os.path.exists(images_path):
        print(f"ERROR: {images_path} not found")
        sys.exit(1)

    n_images = len([f for f in os.listdir(images_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Images:  {n_images} in images_2/")

    all_results = {}

    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        print(f"\n{'='*60}")
        print(f"CONFIG {cfg_name}: {cfg['description']}")
        print(f"{'='*60}")

        results = {"config": cfg_name, "description": cfg["description"]}

        if not args.brush_only:
            print(f"\n--- vksplat ---")
            results["vksplat"] = run_vksplat(
                args.dataset_dir, cfg_name, cfg["vksplat"], args.output_base)

        if not args.vksplat_only:
            print(f"\n--- Brush ---")
            results["brush"] = run_brush(
                args.dataset_dir, cfg_name, cfg["brush"], args.output_base)

        all_results[cfg_name] = results

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    header = f"{'Config':<8} {'Trainer':<10} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'Time':>8} {'Splats':>10}"
    print(header)
    print("-" * len(header))

    for cfg_name, results in all_results.items():
        if "vksplat" in results:
            v = results["vksplat"]
            psnr = v.get("psnr", 0)
            ssim = v.get("ssim", 0)
            lpips = v.get("lpips_alex", v.get("lpips", 0))
            elapsed = v.get("elapsed", 0)
            print(f"{cfg_name:<8} {'vksplat':<10} {psnr:>7.2f} {ssim:>7.3f} "
                  f"{lpips:>7.3f} {elapsed:>7.0f}s")

        if "brush" in results:
            b = results["brush"]
            if "error" not in b:
                psnr = b.get("psnr", 0)
                ssim = b.get("ssim", 0)
                lpips = b.get("lpips_alex", 0)
                elapsed = b.get("elapsed", 0)
                psnr_s = f"{psnr:>7.2f}" if psnr else f"{'n/a':>7}"
                ssim_s = f"{ssim:>7.3f}" if ssim else f"{'n/a':>7}"
                lpips_s = f"{lpips:>7.3f}" if lpips else f"{'n/a':>7}"
                print(f"{cfg_name:<8} {'Brush':<10} {psnr_s} {ssim_s} "
                      f"{lpips_s} {elapsed:>7.0f}s")
            else:
                print(f"{cfg_name:<8} {'Brush':<10} ERROR: {b['error']}")

    # Save full results
    results_path = os.path.join(args.output_base, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    main()
