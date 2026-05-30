"""Video-to-splat pipeline v3 — sharp-frames + multi-video + high-res training.

Changes from v2:
1. Frame selection via sharp-frames-python (best-n / outlier-removal)
2. Multi-video input with proportional frame budgets
3. ALIKED removed — SIFT-only with 16384 features
4. COLMAP overlap default 50 with auto-retry on low registration
5. Training on images_2 (1920px) instead of images_4 (540px)
6. 12 workers default throughout
7. 100K steps, cap 2M, best-of-3 defaults

Usage:
  python video_to_splat_v2.py E:\\Downloads\\vid1.mp4 E:\\Downloads\\vid2.mp4 --name kitchen
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: Probe
# ---------------------------------------------------------------------------

def probe_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    info = {
        "path": video_path,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["total_frames"] / info["fps"]
    cap.release()
    return info


# ---------------------------------------------------------------------------
# Stage 2-3: sharp-frames extraction + resize tiers
# ---------------------------------------------------------------------------

def run_sharp_frames(video_path: str, output_dir: str, num_frames: int,
                     selection_method: str = "best-n",
                     fps: int = 10, min_buffer: int = 3,
                     outlier_sensitivity: int = 50) -> dict:
    """Run sharp-frames on a single video via Python API, return metadata."""
    from sharp_frames import SharpFrames

    os.makedirs(output_dir, exist_ok=True)

    print(f"    sharp-frames: {os.path.basename(video_path)} -> {output_dir} "
          f"(method={selection_method}, target={num_frames}, fps={fps})")
    sys.stdout.flush()

    sf = SharpFrames(
        input_path=video_path,
        input_type="video",
        output_dir=output_dir,
        fps=fps,
        num_frames=num_frames,
        min_buffer=min_buffer,
        output_format="jpg",
        force_overwrite=True,
        selection_method=selection_method,
        outlier_sensitivity=outlier_sensitivity,
    )
    success = sf.run()
    if not success:
        raise RuntimeError("sharp-frames processing failed")

    metadata_path = os.path.join(output_dir, "selected_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            return json.load(f)
    n_out = len([f for f in os.listdir(output_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return {"n_selected": n_out}


def extract_and_resize(videos: list[dict], out_root: str,
                       num_frames: int, width_1: int, width_2: int,
                       width_4: int, selection_method: str = "best-n",
                       fps: int = 10, min_buffer: int = 3,
                       min_frames: int = 450) -> int:
    """Extract sharp frames from one or more videos and create resized tiers.

    Returns total number of frames written.
    """
    from prepare_dataset import save_jpeg

    tiers = [("images_2", width_2), ("images_4", width_4)]
    if width_1 and width_1 > width_2:
        tiers.insert(0, ("images_1", width_1))
    dirs = {name: os.path.join(out_root, name) for name, _ in tiers}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    total_duration = sum(v["duration"] for v in videos)
    written = 0

    for vi, vinfo in enumerate(videos):
        # Proportional frame budget
        frac = vinfo["duration"] / total_duration
        budget = max(30, int(num_frames * frac))
        prefix = f"v{vi}" if len(videos) > 1 else ""

        print(f"\n  Video {vi}: {os.path.basename(vinfo['path'])} "
              f"({vinfo['duration']:.0f}s, budget={budget} frames)")

        sf_dir = os.path.join(out_root, f"_sharp_frames_{vi}")
        metadata = run_sharp_frames(
            vinfo["path"], sf_dir, num_frames=budget,
            selection_method=selection_method, fps=fps,
            min_buffer=min_buffer,
        )

        extracted = sorted([f for f in os.listdir(sf_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"    sharp-frames extracted {len(extracted)} frames")

        for fname in extracted:
            src = os.path.join(sf_dir, fname)
            img = cv2.imread(src)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            stem_base = f"{prefix}_frame_{written:05d}.jpg" if prefix else f"frame_{written:05d}.jpg"

            for tier, tw in tiers:
                path = os.path.join(dirs[tier], stem_base)
                if tw >= w:
                    out = rgb
                else:
                    th_out = int(h * tw / w)
                    out = cv2.resize(rgb, (tw, th_out), interpolation=cv2.INTER_AREA)
                save_jpeg(path, out)

            written += 1

        # Cleanup temp dir
        shutil.rmtree(sf_dir, ignore_errors=True)
        meta_file = os.path.join(sf_dir, "selected_metadata.json")
        if os.path.exists(meta_file):
            os.remove(meta_file)

    if written < min_frames:
        print(f"  WARNING: Only {written} frames extracted (minimum target: {min_frames})")

    return written


def write_every_nth_frame(video_path: str, every_n: int,
                          out_root: str, width_2: int, width_4: int) -> int:
    """Decode video and write every Nth frame. No scoring, no dedup."""
    from prepare_dataset import save_jpeg

    dirs = {
        "images_2": os.path.join(out_root, "images_2"),
        "images_4": os.path.join(out_root, "images_4"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    frame_idx = 0
    t0 = time.time()
    last_print = t0

    print(f"  Writing every {every_n}th frame (~{total // every_n} expected)")
    sys.stdout.flush()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            stem = f"frame_{written:05d}.jpg"

            for tier, tw in [("images_2", width_2), ("images_4", width_4)]:
                path = os.path.join(dirs[tier], stem)
                if tw >= w:
                    out = rgb
                else:
                    th_out = int(h * tw / w)
                    out = cv2.resize(rgb, (tw, th_out), interpolation=cv2.INTER_AREA)
                save_jpeg(path, out)

            written += 1

            now = time.time()
            if now - last_print > 10.0 or written == 1:
                pct = 100.0 * frame_idx / max(total, 1)
                print(f"    [{pct:.0f}%] written={written}", flush=True)
                last_print = now

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"  Write done: {written} frames in {elapsed:.0f}s")
    return written


# ---------------------------------------------------------------------------
# Stage 4: COLMAP (SIFT-only, with retry)
# ---------------------------------------------------------------------------

def run_colmap(dataset_dir: str, image_subdir: str = "images_2",
               camera_params: str = None, num_threads: int = 12,
               overlap: int = 50, use_exhaustive: bool = False) -> dict:
    """Run COLMAP with sequential + optional exhaustive matching. SIFT-only."""
    import pycolmap

    image_dir = os.path.join(dataset_dir, image_subdir)
    db_path = os.path.join(dataset_dir, "database.db")
    sparse_dir = os.path.join(dataset_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)

    n_images = len([f for f in os.listdir(image_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    match_mode = "sequential+exhaustive" if use_exhaustive else "sequential"
    print(f"  COLMAP on {n_images} images in {image_subdir} "
          f"({num_threads} threads, overlap={overlap}, matching={match_mode})")

    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "SIMPLE_RADIAL"
    if camera_params:
        reader_opts.camera_params = camera_params
        print(f"  Primed camera: {camera_params}")

    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.num_threads = num_threads
    extraction_opts.type = pycolmap.FeatureExtractorType.SIFT
    extraction_opts.sift.first_octave = 0
    extraction_opts.sift.max_num_features = 16384
    print(f"  Features: SIFT (max_features=16384)")

    matching_opts = pycolmap.FeatureMatchingOptions()
    matching_opts.num_threads = num_threads

    t0 = time.time()
    print("  Step 1/4: Feature extraction...")
    sys.stdout.flush()
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extraction_opts,
        device=pycolmap.Device.cpu,
    )
    print(f"    Done in {time.time()-t0:.0f}s")

    t1 = time.time()
    print(f"  Step 2/4: Sequential matching (overlap={overlap})...")
    sys.stdout.flush()
    pairing_opts = pycolmap.SequentialPairingOptions()
    pairing_opts.overlap = overlap
    pairing_opts.quadratic_overlap = True
    pairing_opts.num_threads = num_threads

    pycolmap.match_sequential(
        database_path=db_path,
        matching_options=matching_opts,
        pairing_options=pairing_opts,
        device=pycolmap.Device.cpu,
    )
    print(f"    Done in {time.time()-t1:.0f}s")

    if use_exhaustive:
        t1b = time.time()
        print(f"  Step 3/4: Exhaustive matching (bridging video segments)...")
        sys.stdout.flush()
        pycolmap.match_exhaustive(
            database_path=db_path,
            matching_options=matching_opts,
            device=pycolmap.Device.cpu,
        )
        print(f"    Done in {time.time()-t1b:.0f}s")
    else:
        print(f"  Step 3/4: (exhaustive matching skipped)")

    t2 = time.time()
    print("  Step 4/4: Incremental mapping...")
    sys.stdout.flush()
    pipeline_opts = pycolmap.IncrementalPipelineOptions()
    pipeline_opts.multiple_models = False
    pipeline_opts.ba_refine_principal_point = False
    pipeline_opts.ba_refine_extra_params = False
    pipeline_opts.ba_refine_focal_length = True
    pipeline_opts.num_threads = num_threads

    _reg_count = [0]
    _last_log = [time.time()]

    def _on_next_image():
        _reg_count[0] += 1
        now = time.time()
        if now - _last_log[0] >= 10.0 or _reg_count[0] <= 3:
            elapsed = now - t2
            rate = _reg_count[0] / max(elapsed, 1) * 60
            print(f"    [{_reg_count[0]}/{n_images}] registered "
                  f"({elapsed:.0f}s elapsed, {rate:.1f} img/min)", flush=True)
            _last_log[0] = now

    def _on_initial_pair():
        elapsed = time.time() - t2
        print(f"    Initial pair found ({elapsed:.0f}s)", flush=True)

    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=sparse_dir,
        options=pipeline_opts,
        initial_image_pair_callback=_on_initial_pair,
        next_image_callback=_on_next_image,
    )
    print(f"    Done in {time.time()-t2:.0f}s "
          f"({_reg_count[0]} images registered)")
    print(f"  Total COLMAP: {time.time()-t0:.0f}s")

    result = {"n_input": n_images, "models": []}
    for idx, recon in reconstructions.items():
        n_reg = recon.num_reg_images()
        n_pts = recon.num_points3D()
        cam = list(recon.cameras.values())[0]
        info = {
            "model_idx": idx,
            "registered": n_reg,
            "points3D": n_pts,
            "camera": (f"{cam.model_name} {cam.width}x{cam.height} "
                       f"f={cam.params[0]:.1f} k={cam.params[3]:.4f}"),
        }
        result["models"].append(info)
        print(f"  Model {idx}: {n_reg}/{n_images} registered, {n_pts} pts, "
              f"f={cam.params[0]:.1f}")
    return result


def run_colmap_with_retry(dataset_dir: str, image_subdir: str,
                          camera_params: str, num_threads: int,
                          initial_overlap: int, max_retries: int = 2,
                          min_reg_rate: float = 0.7) -> dict:
    """Run COLMAP with automatic retry: first increases overlap, then adds
    exhaustive matching to bridge disconnected video segments."""
    overlap = initial_overlap
    use_exhaustive = False
    for attempt in range(1 + max_retries):
        if attempt > 0:
            strategy = f"overlap={overlap}"
            if use_exhaustive:
                strategy += " + exhaustive matching"
            print(f"\n  COLMAP retry {attempt}/{max_retries} ({strategy})...")

        result = run_colmap(
            dataset_dir, image_subdir, camera_params,
            num_threads=num_threads, overlap=overlap,
            use_exhaustive=use_exhaustive,
        )

        if not result["models"]:
            print("  ERROR: COLMAP produced no reconstruction!")
            if attempt < max_retries:
                use_exhaustive = True
                continue
            return result

        best_model = max(result["models"], key=lambda m: m["registered"])
        reg_rate = best_model["registered"] / result["n_input"]
        print(f"  Registration rate: {reg_rate*100:.1f}%")

        if reg_rate >= min_reg_rate:
            return result

        if attempt < max_retries:
            print(f"  Registration {reg_rate*100:.0f}% < {min_reg_rate*100:.0f}% target, retrying...")
            use_exhaustive = True
        else:
            print(f"  WARNING: Registration {reg_rate*100:.0f}% still below target after retries.")

    return result


# ---------------------------------------------------------------------------
# Stage 5: Post-COLMAP filtering
# ---------------------------------------------------------------------------

def filter_colmap(dataset_dir: str, min_points_factor: float = 0.1) -> dict:
    """Remove poorly-registered images from sparse/0."""
    import pycolmap

    sparse_path = os.path.join(dataset_dir, "sparse", "0")
    if not os.path.exists(sparse_path):
        print("  WARNING: sparse/0 not found, skipping filter")
        return {"filtered": 0, "remaining": 0}

    recon = pycolmap.Reconstruction(sparse_path)
    n_before = recon.num_reg_images()

    pts_per_img = {}
    for img_id, img in recon.images.items():
        n_pts = sum(1 for p in img.points2D if p.has_point3D())
        pts_per_img[img_id] = n_pts

    if not pts_per_img:
        return {"filtered": 0, "remaining": n_before}

    median_pts = np.median(list(pts_per_img.values()))
    threshold = max(10, median_pts * min_points_factor)

    bad_ids = [k for k, v in pts_per_img.items() if v < threshold]
    for img_id in bad_ids:
        img = recon.images[img_id]
        recon.deregister_frame(img.frame_id)

    bad_pts = [pid for pid, pt in recon.points3D.items()
               if pt.track.length() < 3]
    for pid in bad_pts:
        recon.delete_point3D(pid)

    if bad_ids or bad_pts:
        recon.write(sparse_path)

    n_after = recon.num_reg_images()
    print(f"  Filter: {n_before} -> {n_after} images "
          f"(removed {len(bad_ids)} with <{threshold:.0f} triangulated pts)")
    return {"filtered": len(bad_ids), "remaining": n_after}


# ---------------------------------------------------------------------------
# Stage 6: Training (Brush)
# ---------------------------------------------------------------------------

BRUSH_EXE = r"E:\brush\target\release\brush.exe"


def train_brush(dataset_dir: str, output_base: str, name: str,
                steps: int = 30000, max_splats: int = 1000000,
                ssim_weight: float = 0.2, max_resolution: int = 960,
                n_runs: int = 1) -> list:
    """Train with Brush (Rust/WebGPU 3DGS trainer)."""
    outputs = []

    for i in range(n_runs):
        export_dir = os.path.join(output_base, f"{name}_run{i}")
        os.makedirs(export_dir, exist_ok=True)

        print(f"\n  === Brush run {i+1}/{n_runs} ===")
        print(f"    steps={steps}, splats={max_splats}, ssim={ssim_weight}, "
              f"res={max_resolution}")
        sys.stdout.flush()

        cmd = [
            BRUSH_EXE, dataset_dir,
            "--total-train-iters", str(steps),
            "--max-splats", str(max_splats),
            "--ssim-weight", str(ssim_weight),
            "--max-resolution", str(max_resolution),
            "--export-every", str(steps),
            "--export-path", export_dir,
            "--eval-split-every", "8",
            "--eval-every", str(steps),
            "--eval-save-to-disk",
            "--seed", str(42 + i),
        ]

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        outputs.append({
            "run": i, "export_dir": export_dir,
            "elapsed": elapsed, "returncode": result.returncode,
        })
        print(f"  Run {i+1} done in {elapsed:.0f}s (rc={result.returncode})")

    return outputs


# ---------------------------------------------------------------------------
# Stage 7: Evaluate (with LPIPS)
# ---------------------------------------------------------------------------

def compute_eval_metrics(eval_render_dir: str, gt_image_dir: str) -> dict:
    """Compute PSNR, SSIM, and LPIPS from Brush eval renders vs ground truth."""
    import cv2
    import lpips
    import torch
    from skimage.metrics import structural_similarity

    renders = sorted([f for f in os.listdir(eval_render_dir)
                      if f.lower().endswith(('.png', '.jpg'))])
    if not renders:
        return {}

    lpips_fn = lpips.LPIPS(net='alex', verbose=False)
    if torch.cuda.is_available():
        lpips_fn = lpips_fn.cuda()

    psnrs, ssims, lpips_vals = [], [], []
    for render_name in renders:
        render = cv2.imread(os.path.join(eval_render_dir, render_name))
        if render is None:
            continue

        gt_name = render_name[:-4] if ".jpg.png" in render_name else render_name
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

        ssims.append(structural_similarity(render, gt, channel_axis=2,
                                           data_range=255))

        # LPIPS expects [-1, 1] RGB tensors in CHW format
        r_t = torch.from_numpy(render[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        g_t = torch.from_numpy(gt[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        if torch.cuda.is_available():
            r_t, g_t = r_t.cuda(), g_t.cuda()
        with torch.no_grad():
            lp = lpips_fn(r_t.unsqueeze(0), g_t.unsqueeze(0)).item()
        lpips_vals.append(lp)

    if not psnrs:
        return {}

    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips_alex": float(np.mean(lpips_vals)),
        "n_eval": len(psnrs),
    }


def evaluate_brush_runs(outputs: list, dataset_dir: str,
                        max_resolution: int) -> dict:
    """Evaluate Brush training runs and pick the best by LPIPS."""
    gt_dir = os.path.join(dataset_dir, "images_2")
    if max_resolution <= 960:
        gt_dir_candidate = os.path.join(dataset_dir, "images_4")
        if os.path.isdir(gt_dir_candidate):
            gt_dir = gt_dir_candidate

    results = []
    for run in outputs:
        export_dir = run.get("export_dir")
        if not export_dir or not os.path.isdir(export_dir):
            continue

        eval_dirs = sorted([d for d in os.listdir(export_dir)
                            if d.startswith("eval_") and
                            os.path.isdir(os.path.join(export_dir, d))],
                           key=lambda d: int(d.split("_")[1]))
        if not eval_dirs:
            print(f"  Run {run['run']}: no eval renders found")
            continue

        eval_dir = os.path.join(export_dir, eval_dirs[-1])
        print(f"\n  Run {run['run']}: evaluating from {eval_dirs[-1]}/...")
        metrics = compute_eval_metrics(eval_dir, gt_dir)
        if metrics:
            results.append({"run": run["run"], "dir": export_dir,
                            "eval": metrics})
            print(f"    PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f} "
                  f"LPIPS={metrics['lpips_alex']:.4f} ({metrics['n_eval']} images)")

            eval_path = os.path.join(export_dir, "eval.json")
            with open(eval_path, "w") as f:
                json.dump(metrics, f, indent=2)

    if not results:
        print("  No eval results found!")
        return {}

    best = min(results, key=lambda r: r["eval"]["lpips_alex"])
    print(f"\n  WINNER: Run {best['run']} "
          f"(LPIPS={best['eval']['lpips_alex']:.4f}, "
          f"PSNR={best['eval']['psnr']:.2f})")
    print(f"  Output: {best['dir']}")
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Video to Splat pipeline v3 (sharp-frames + multi-video + high-res)")
    p.add_argument("videos", nargs="+", help="One or more input video paths")
    p.add_argument("--name", default="video_splat_v3")
    p.add_argument("--out-root", default=None)
    p.add_argument("--output-base", default=r"E:\vksplat_output")

    # Frame selection
    p.add_argument("--num-frames", type=int, default=500,
                   help="Target number of frames to extract (total across all videos)")
    p.add_argument("--min-frames", type=int, default=450,
                   help="Warn if fewer than this many frames are extracted")
    p.add_argument("--select-mode", default="sharp",
                   choices=["sharp", "sharp-outlier", "every"],
                   help="Frame selection: sharp (best-n via sharp-frames), "
                        "sharp-outlier (outlier-removal), every (every Nth)")
    p.add_argument("--sharp-fps", type=int, default=10,
                   help="FPS for sharp-frames video extraction")
    p.add_argument("--min-buffer", type=int, default=3,
                   help="Minimum buffer between sharp-frames selected frames")
    p.add_argument("--every-n", type=int, default=10,
                   help="For --select-mode every: take every Nth frame")

    # Resolution
    p.add_argument("--width-1", type=int, default=None,
                   help="Width for images_1 tier (native/high-res, default: video width)")
    p.add_argument("--width-2", type=int, default=1920,
                   help="Width for images_2 tier (COLMAP + default training, 1920)")
    p.add_argument("--width-4", type=int, default=960,
                   help="Width for images_4 tier (fast/fallback, 960)")

    # COLMAP
    p.add_argument("--camera-params", default=None)
    p.add_argument("--overlap", type=int, default=50,
                   help="COLMAP sequential matching overlap (default 50)")
    p.add_argument("--workers", type=int, default=12,
                   help="Thread count for COLMAP and parallel operations")

    # Training (Brush)
    p.add_argument("--steps", type=int, default=30000,
                   help="Brush training iterations (default 30000)")
    p.add_argument("--max-splats", type=int, default=1000000,
                   help="Max splat count (default 1M)")
    p.add_argument("--ssim-weight", type=float, default=0.2,
                   help="SSIM loss weight (default 0.2)")
    p.add_argument("--max-resolution", type=int, default=None,
                   help="Max image dimension for Brush (default: width_2)")
    p.add_argument("--n-runs", type=int, default=1,
                   help="Number of training runs (default 1)")

    # Control flow
    p.add_argument("--check", action="store_true",
                   help="Only extract frames + report stats (no COLMAP/train)")
    p.add_argument("--skip-to", choices=["colmap", "train", "eval"])
    p.add_argument("--no-train", action="store_true",
                   help="Stop after COLMAP (don't train)")

    args = p.parse_args()

    if args.out_root is None:
        args.out_root = os.path.join(r"E:\vksplat_data", args.name)

    print("=" * 60)
    print("VIDEO-TO-SPLAT PIPELINE v3")
    print("=" * 60)

    # --- Stage 1: Probe all videos ---
    print("\n[Stage 1] Probing videos...")
    video_infos = []
    for vp in args.videos:
        info = probe_video(vp)
        video_infos.append(info)
        print(f"  {os.path.basename(vp)}: "
              f"{info['width']}x{info['height']} @ {info['fps']:.1f}fps, "
              f"{info['duration']:.1f}s, {info['total_frames']} frames")

    total_duration = sum(v["duration"] for v in video_infos)
    total_frames = sum(v["total_frames"] for v in video_infos)
    print(f"  Total: {len(video_infos)} videos, {total_duration:.0f}s, "
          f"{total_frames} frames")

    if args.skip_to in ("colmap", "train", "eval"):
        print(f"  Skipping to {args.skip_to}...")

    elif args.select_mode == "every":
        print(f"\n[Stage 2-3] Writing every {args.every_n}th frame (no filtering)...")
        os.makedirs(args.out_root, exist_ok=True)
        n_written = 0
        for vi, vinfo in enumerate(video_infos):
            print(f"\n  Video {vi}: {os.path.basename(vinfo['path'])}")
            n = write_every_nth_frame(
                vinfo["path"], args.every_n, args.out_root,
                args.width_2, args.width_4,
            )
            n_written += n

        sel_info = {
            "select_mode": "every",
            "every_n": args.every_n,
            "total_source_frames": total_frames,
            "n_written": n_written,
            "n_videos": len(video_infos),
        }
        with open(os.path.join(args.out_root, "_selection_info.json"), "w") as f:
            json.dump(sel_info, f, indent=2)
        print(f"\n  Final: {n_written} frames staged in {args.out_root}")

    else:
        # --- sharp-frames selection ---
        sf_method = "best-n" if args.select_mode == "sharp" else "outlier-removal"
        print(f"\n[Stage 2-3] sharp-frames extraction "
              f"(method={sf_method}, target={args.num_frames} frames)...")

        os.makedirs(args.out_root, exist_ok=True)
        width_1 = args.width_1 or video_infos[0]["width"]
        n_written = extract_and_resize(
            video_infos, args.out_root,
            num_frames=args.num_frames,
            width_1=width_1, width_2=args.width_2, width_4=args.width_4,
            selection_method=sf_method,
            fps=args.sharp_fps, min_buffer=args.min_buffer,
            min_frames=args.min_frames,
        )

        sel_info = {
            "select_mode": args.select_mode,
            "sharp_frames_method": sf_method,
            "target_frames": args.num_frames,
            "n_written": n_written,
            "n_videos": len(video_infos),
            "total_source_frames": total_frames,
            "width_2": args.width_2,
            "width_4": args.width_4,
        }
        with open(os.path.join(args.out_root, "_selection_info.json"), "w") as f:
            json.dump(sel_info, f, indent=2)
        print(f"\n  Final: {n_written} frames staged in {args.out_root}")

    if args.check:
        print("\n--check mode: stopping here.")
        return

    if args.skip_to == "eval":
        print("  Skipping to eval...")
    elif args.skip_to == "train":
        print("  Skipping to train...")
    else:
        # --- Stage 4: COLMAP ---
        print(f"\n[Stage 4] COLMAP...")
        cam_params = args.camera_params
        if cam_params is None:
            print(f"  No camera params specified — COLMAP will estimate intrinsics")

        colmap_result = run_colmap_with_retry(
            args.out_root, "images_2", cam_params,
            num_threads=args.workers,
            initial_overlap=args.overlap,
        )

        if not colmap_result["models"]:
            print("  ERROR: COLMAP produced no reconstruction!")
            return

        # --- Stage 5: Filter ---
        print(f"\n[Stage 5] Post-COLMAP filtering...")
        filter_result = filter_colmap(args.out_root)

    if args.no_train:
        print("\n--no-train: stopping after COLMAP.")
        return

    max_res = args.max_resolution or args.width_2

    if args.skip_to != "eval":
        # --- Stage 6: Train (Brush) ---
        print(f"\n[Stage 6] Brush training ({args.n_runs} run(s), "
              f"res={max_res}, steps={args.steps}, splats={args.max_splats})...")
        outputs = train_brush(
            args.out_root, args.output_base, args.name,
            steps=args.steps, max_splats=args.max_splats,
            ssim_weight=args.ssim_weight, max_resolution=max_res,
            n_runs=args.n_runs,
        )
    else:
        outputs = []

    # --- Stage 7: Evaluate (PSNR + SSIM + LPIPS) ---
    print(f"\n[Stage 7] Evaluation (PSNR, SSIM, LPIPS-Alex)...")
    winner = evaluate_brush_runs(outputs, args.out_root, max_res)

    print("\n" + "=" * 60)
    print("PIPELINE v3 COMPLETE")
    print("=" * 60)
    if winner:
        ev = winner.get("eval", {})
        print(f"Best splat: {winner.get('dir', 'unknown')}")
        print(f"  LPIPS={ev.get('lpips_alex', 'n/a'):.4f}  "
              f"PSNR={ev.get('psnr', 'n/a'):.2f}  "
              f"SSIM={ev.get('ssim', 'n/a'):.3f}")


if __name__ == "__main__":
    main()
