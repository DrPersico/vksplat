"""Select the sharpest frames from a video (or photo folder) and stage them.

Why this exists: the original 900-photo dataset AND the 4K60 video are both
heavily motion-blurred (variance-of-Laplacian ~15; sharp images score 500+).
This tool extracts candidate frames, scores each by sharpness, and keeps only
the locally-sharpest frames while preserving scene coverage (temporal buckets).

Two modes:
  --video  <mp4>   : ffmpeg-extract candidates, then select
  --photos <dir>   : score an existing folder of images, then select

Sharpness metric: variance of the Laplacian (same metric used to diagnose the
blur problem). Higher = sharper. Rough guide: >500 sharp, 150-500 usable,
<150 too blurry for detailed reconstruction.

Selection: split the candidate sequence into N temporal buckets; from each
bucket keep the top-K sharpest. Guarantees every part of the walk-through is
represented (coverage) while discarding blurry frames (quality).

Output: staged into <out_root>/{images_2,images_4}/ via the SAME resize +
JPEG-strip path as prepare_dataset.py (helpers imported, not duplicated).

Usage:
  python select_sharp_frames.py --video E:\\Downloads\\clip.mp4 --out-root E:\\vksplat_data\\sharp --target-count 400
  python select_sharp_frames.py --photos "E:\\Downloads\\50mb full dataset" --check
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

# Reuse the proven resize + EXIF-strip primitives — do not reimplement.
from prepare_dataset import save_jpeg, apply_orientation, EXIF_ORIENTATION_TAG
from PIL import Image


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_gray(path: str) -> np.ndarray:
    """Windows-path-safe grayscale read."""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)


def read_color(path: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def extract_video_candidates(video: str, work_dir: str, fps: float,
                             start: float, duration: float) -> list:
    """ffmpeg-extract candidate frames at `fps`. Returns sorted file list.

    HEVC decode confirmed working. Output to a Windows path (bash /tmp fails).
    """
    os.makedirs(work_dir, exist_ok=True)
    for old in glob.glob(os.path.join(work_dir, "cand_*.jpg")):
        os.remove(old)
    cmd = ["ffmpeg", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video]
    if duration > 0:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", f"fps={fps}", "-q:v", "2",
            os.path.join(work_dir, "cand_%06d.jpg")]
    print(f"Extracting candidates @ {fps} fps ...")
    sys.stdout.flush()
    t0 = time.time()
    subprocess.run(cmd, check=True)
    files = sorted(glob.glob(os.path.join(work_dir, "cand_*.jpg")))
    print(f"  {len(files)} candidate frames in {time.time()-t0:.0f}s")
    return files


def score_all(files: list) -> np.ndarray:
    """Score every candidate. Returns float array aligned with `files`."""
    scores = np.empty(len(files), np.float64)
    t0 = time.time()
    for i, f in enumerate(files):
        g = read_gray(f)
        scores[i] = sharpness(g) if g is not None else 0.0
        if i == 0 or time.time() - t0 > 3.0 or i == len(files) - 1:
            print(f"  scored [{i+1}/{len(files)}]")
            sys.stdout.flush()
            t0 = time.time()
    return scores


def report(scores: np.ndarray) -> None:
    a = scores
    print("\n=== Sharpness distribution (variance of Laplacian) ===")
    print(f"  count={len(a)}  min={a.min():.0f}  median={np.median(a):.0f}  "
          f"mean={a.mean():.0f}  max={a.max():.0f}")
    for p in (50, 75, 90, 95, 99):
        print(f"  p{p}={np.percentile(a, p):.0f}")
    for thr in (100, 150, 300, 500):
        n = int((a > thr).sum())
        print(f"  frames > {thr}: {n} ({100*n/len(a):.1f}%)")
    med = np.median(a)
    if med < 150:
        print(f"\n  WARNING: median {med:.0f} < 150 — capture is too blurry "
              f"for detailed reconstruction. Recapture recommended.")
    else:
        print(f"\n  OK: median {med:.0f} is usable.")

    # Temporal coverage: are usable frames spread across the walk-through
    # (reconstructable) or clustered in one spot (useless for 3D)?
    n_seg = 12
    edges = np.linspace(0, len(a), n_seg + 1, dtype=int)
    print(f"\n  Temporal coverage ({n_seg} segments, # frames > 100 each):")
    covered = 0
    for s in range(n_seg):
        lo, hi = edges[s], edges[s + 1]
        seg = a[lo:hi]
        ngood = int((seg > 100).sum())
        if ngood > 0:
            covered += 1
        bar = "#" * min(ngood, 40)
        print(f"   seg {s:2d}: {ngood:4d} >100  best={seg.max():3.0f}  {bar}")
    print(f"  segments with >=1 usable frame: {covered}/{n_seg} "
          f"({'good spread' if covered >= n_seg*0.7 else 'POOR — clustered, weak for COLMAP'})")


def select_bucketed(scores: np.ndarray, target: int) -> np.ndarray:
    """Return indices of selected frames: top-K sharpest per temporal bucket.

    Buckets preserve scene coverage; per-bucket top-K enforces quality.
    """
    n = len(scores)
    if target >= n:
        return np.arange(n)
    n_buckets = min(target, n)
    edges = np.linspace(0, n, n_buckets + 1, dtype=int)
    per = max(1, target // n_buckets)
    chosen = []
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        local = np.arange(lo, hi)
        k = min(per, len(local))
        top = local[np.argsort(scores[local])[-k:]]
        chosen.extend(top.tolist())
    return np.array(sorted(set(chosen)))


def stage(files: list, idxs: np.ndarray, out_root: str,
          width_2: int, width_4: int, from_video: bool) -> None:
    tiers = {"images_2": width_2, "images_4": width_4}
    for t in tiers:
        os.makedirs(os.path.join(out_root, t), exist_ok=True)
    print(f"\nStaging {len(idxs)} frames -> {out_root}\\{{images_2,images_4}}")
    for n, i in enumerate(idxs):
        src = files[i]
        if from_video:
            # ffmpeg output: no EXIF, orientation already baked in.
            arr = cv2.cvtColor(read_color(src), cv2.COLOR_BGR2RGB)
        else:
            with Image.open(src) as im:
                orient = im.getexif().get(EXIF_ORIENTATION_TAG, 1)
                arr = np.array(im)
            arr = apply_orientation(arr, orient)
        oh, ow = arr.shape[:2]
        stem = f"sharp_{n:05d}.jpg"
        for tier, tw in tiers.items():
            if tw == 0 or tw >= ow:
                out = arr
            else:
                th = round(oh * tw / ow)
                out = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)
            save_jpeg(os.path.join(out_root, tier, stem), out)
        if n == 0 or (n + 1) % 50 == 0 or n == len(idxs) - 1:
            print(f"  [{n+1}/{len(idxs)}] {stem} {ow}x{oh}")
            sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="Source video (mp4/hevc)")
    g.add_argument("--photos", help="Source folder of images")
    p.add_argument("--out-root", help="Dataset root to stage into")
    p.add_argument("--target-count", type=int, default=400,
                   help="Approx number of sharpest frames to keep (default 400)")
    p.add_argument("--fps", type=float, default=4.0,
                   help="Video: candidate extraction rate (default 4 fps)")
    p.add_argument("--start", type=float, default=0.0, help="Video start sec")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Video seconds to use (0 = all)")
    p.add_argument("--width-2", type=int, default=2040)
    p.add_argument("--width-4", type=int, default=1020)
    p.add_argument("--check", action="store_true",
                   help="Only score + report distribution, do not stage")
    p.add_argument("--work-dir", default=r"E:\vksplat_data\_frame_candidates")
    args = p.parse_args()

    if args.video:
        files = extract_video_candidates(args.video, args.work_dir,
                                         args.fps, args.start, args.duration)
        from_video = True
    else:
        pats = ("*.jpg", "*.jpeg", "*.JPG", "*.png", "*.PNG")
        files = []
        for pat in pats:
            files.extend(glob.glob(os.path.join(args.photos, pat)))
        files = sorted(set(files))
        from_video = False
        print(f"{len(files)} source images in {args.photos}")

    if not files:
        print("No frames found.", file=sys.stderr)
        sys.exit(1)

    scores = score_all(files)
    report(scores)

    if args.check:
        print("\n--check: scoring only, nothing staged.")
        return

    if not args.out_root:
        print("ERROR: --out-root required unless --check", file=sys.stderr)
        sys.exit(1)

    idxs = select_bucketed(scores, args.target_count)
    sel = scores[idxs]
    print(f"\nSelected {len(idxs)} frames. "
          f"Selected-set sharpness: min={sel.min():.0f} "
          f"median={np.median(sel):.0f} max={sel.max():.0f}")
    stage(files, idxs, args.out_root, args.width_2, args.width_4, from_video)
    print("\nDone. Next: run_colmap_sequential.py on", args.out_root)


if __name__ == "__main__":
    main()
