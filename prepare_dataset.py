"""Stage a flat folder of source JPEGs into vksplat's dataset layout.

For each source image:
  1. Load raw pixels via PIL (ignores EXIF orientation — critical, see memory.md issue #1)
  2. Apply EXIF orientation manually so the saved pixels are visually upright
     AND so COLMAP/training agree on pixel coordinates (we strip EXIF on write).
  3. Resize with OpenCV (INTER_AREA) to each target width.
  4. Save as JPEG quality 95 with NO EXIF metadata.

Output layout (mirrors E:\\vksplat_data\\livingroom):
  <out_root>/images/    source resolution (post-orientation, no resize)
  <out_root>/images_2/  resized to --width-2 (default 2040 — matches livingroom's ~2000-wide)
  <out_root>/images_4/  resized to --width-4 (default 1020)

The "_2" and "_4" suffixes refer to the vksplat convention (images_2 ~ HD-ish),
NOT literal fractions of source. Adjust widths to match your source resolution.

Usage:
  python prepare_dataset.py <src_dir> <out_root> [--skip-fullres] [--max N]
"""

import argparse
import os
import sys
import glob
import time
import numpy as np
import cv2
from PIL import Image


EXIF_ORIENTATION_TAG = 274


def apply_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply EXIF orientation to raw pixel array. Returns visually-upright pixels.

    EXIF orientations: 1=normal, 2=mirror-h, 3=180, 4=mirror-v,
    5=mirror-h+90CCW, 6=90CW, 7=mirror-h+90CW, 8=90CCW.
    """
    if orientation in (None, 1):
        return arr
    if orientation == 2:
        return arr[:, ::-1]
    if orientation == 3:
        return arr[::-1, ::-1]
    if orientation == 4:
        return arr[::-1, :]
    if orientation == 5:
        return np.rot90(arr[:, ::-1], k=-1)
    if orientation == 6:
        return np.rot90(arr, k=-1)
    if orientation == 7:
        return np.rot90(arr[:, ::-1], k=1)
    if orientation == 8:
        return np.rot90(arr, k=1)
    return arr


def save_jpeg(path: str, rgb: np.ndarray, quality: int = 95) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def process_one(src_path: str, out_paths: dict, target_widths: dict) -> tuple:
    """Returns (orig_w, orig_h, oriented_w, oriented_h).

    target_widths[tier] == 0 means "keep source resolution unchanged".
    """
    ext = os.path.splitext(src_path)[1].lower()
    if ext in ('.arw',):
        import rawpy
        with rawpy.imread(src_path) as raw:
            # rawpy postprocess automatically handles auto-rotation (using EXIF).
            arr = raw.postprocess()
        orig_h, orig_w = arr.shape[:2]
        oriented_h, oriented_w = orig_h, orig_w
    else:
        with Image.open(src_path) as im:
            ex = im.getexif()
            orient = ex.get(EXIF_ORIENTATION_TAG, 1)
            arr = np.array(im)
        orig_h, orig_w = arr.shape[:2]
        arr = apply_orientation(arr, orient)
        oriented_h, oriented_w = arr.shape[:2]

    for tier, out_path in out_paths.items():
        tw = target_widths[tier]
        if tw == 0 or tw >= oriented_w:
            tier_img = arr
        else:
            th = round(oriented_h * tw / oriented_w)
            tier_img = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)
        save_jpeg(out_path, tier_img)

    return orig_w, orig_h, oriented_w, oriented_h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src_dir", help="Source folder of JPEGs")
    p.add_argument("out_root", help="Output dataset root (will create images_*/ subfolders)")
    p.add_argument("--skip-fullres", action="store_true",
                   help="Skip writing the full-resolution images/ tier (saves disk and RAM)")
    p.add_argument("--width-2", type=int, default=2040,
                   help="Target width for images_2/ tier (default 2040, matches livingroom's ~2000-wide)")
    p.add_argument("--width-4", type=int, default=1020,
                   help="Target width for images_4/ tier (default 1020)")
    p.add_argument("--max", type=int, default=None,
                   help="Limit to first N images (for quick testing)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = p.parse_args()

    patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG", "*.arw", "*.ARW")
    files: list = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(args.src_dir, pat)))
    files = sorted(set(files))
    if args.max is not None:
        files = files[: args.max]
    if not files:
        print(f"No images found in {args.src_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} images in {args.src_dir}")

    tiers = {"images_2": args.width_2, "images_4": args.width_4}
    if not args.skip_fullres:
        tiers = {"images": 0, **tiers}

    for tier in tiers:
        os.makedirs(os.path.join(args.out_root, tier), exist_ok=True)
    print(f"Writing tiers: {list(tiers.keys())}")
    print(f"Output root: {args.out_root}")

    t0 = time.time()
    last_print = t0
    for i, src in enumerate(files):
        stem = os.path.splitext(os.path.basename(src))[0] + ".jpg"
        out_paths = {tier: os.path.join(args.out_root, tier, stem) for tier in tiers}
        if all(os.path.exists(p) for p in out_paths.values()):
            continue
        ow, oh, w, h = process_one(src, out_paths, tiers)
        now = time.time()
        if i == 0 or now - last_print > 2.0 or i == len(files) - 1:
            elapsed = now - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(files) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(files)}] {os.path.basename(src)} "
                  f"orig={ow}x{oh} -> oriented={w}x{h}  "
                  f"({rate:.1f} img/s, ETA {eta:.0f}s)")
            last_print = now

    print(f"\nDone in {time.time()-t0:.1f}s")
    for tier in tiers:
        d = os.path.join(args.out_root, tier)
        n = len([f for f in os.listdir(d) if f.lower().endswith(".jpg")])
        sample = next((f for f in os.listdir(d) if f.lower().endswith(".jpg")), None)
        if sample:
            with Image.open(os.path.join(d, sample)) as im:
                print(f"  {tier}/: {n} files, sample size {im.size}")


if __name__ == "__main__":
    main()
