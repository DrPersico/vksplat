"""Run COLMAP SfM via pycolmap on a dataset directory.

Usage: python run_colmap.py <dataset_dir> [--image-subdir images_2]
Expects <dataset_dir>/<image_subdir>/ with JPEG files.
Outputs to <dataset_dir>/sparse/0/

GPU mode (much faster feature extraction + matching on NVIDIA):
  python run_colmap.py <dataset_dir> --image-subdir images_2 \
      --camera-params 1343.6,1000,462,0.051 --gpu

  --gpu shells out to a standalone CUDA-enabled colmap.exe for the two
  O(N^2)/GPU-friendly stages (feature extraction + exhaustive matching).
  COLMAP's incremental mapper is CPU-only upstream, so stage 3 always
  stays on pycolmap. Default (no --gpu) is unchanged: pure pycolmap CPU.
"""

import sys
import os
import time
import argparse
import subprocess
import pycolmap

# Default install location for the CUDA COLMAP binary (kept OFF C: per
# project constraint -- C: is almost full). Override with --colmap-exe.
DEFAULT_COLMAP_EXE = r"E:\vksplat_tools\bin\colmap.exe"


def _run(cmd):
    """Run a subprocess, streaming output, raising on non-zero exit."""
    print("  $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"COLMAP step failed (exit {proc.returncode}): {cmd[1]}")


def gpu_extract_and_match(colmap_exe, db_path, image_dir, args):
    """Stages 1+2 via standalone CUDA colmap.exe (GPU SIFT + matching).

    Mirrors the pycolmap CPU config: SINGLE shared SIMPLE_RADIAL camera,
    first_octave=0 (no upscale), max 8192 features, optional primed
    intrinsics. Mapping (stage 3) stays on pycolmap -- COLMAP's
    incremental mapper has no GPU path upstream.
    """
    extract = [
        colmap_exe, "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--SiftExtraction.first_octave", "0",
        "--SiftExtraction.max_num_features", "8192",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", str(args.gpu_index),
    ]
    if args.camera_params:
        extract += ["--ImageReader.camera_params", args.camera_params]
    print("Step 1/3: GPU feature extraction (CUDA SIFT)...", flush=True)
    _run(extract)

    match = [
        colmap_exe, "exhaustive_matcher",
        "--database_path", db_path,
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", str(args.gpu_index),
    ]
    print("Step 2/3: GPU exhaustive matching...", flush=True)
    _run(match)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir")
    p.add_argument("--image-subdir", default="images_2",
                   help="Subdirectory with images (default: images_2 to save RAM)")
    p.add_argument("--camera-params", default=None,
                   help="Prime SIMPLE_RADIAL intrinsics 'f,cx,cy,k' (fixes "
                        "degenerate auto-estimation; see memory.md issue #4). "
                        "For 2000x924 Samsung: '1343.6,1000,462,0.051'")
    p.add_argument("--no-refine-extra", action="store_true",
                   help="Disable BA refinement of distortion (use with "
                        "--camera-params to keep primed intrinsics fixed)")
    p.add_argument("--num-threads", type=int, default=12,
                   help="CPU threads for all stages. Default 12 = this box's "
                        "logical core count (Ryzen 5 3600, 6c/12t); best for "
                        "Stage 3 incremental mapping/bundle-adjustment, which "
                        "is CPU-bound even in --gpu mode (GPU only covers "
                        "stages 1+2). For very large 4000px+/8K images that "
                        "load a full image per extraction thread, drop to ~4 "
                        "to stay RAM-safe; for <=2000px images 12 is fine.")
    p.add_argument("--gpu", action="store_true",
                   help="Use a standalone CUDA colmap.exe for feature "
                        "extraction + exhaustive matching (NVIDIA only; "
                        "~10-20x faster than CPU on large datasets). "
                        "Mapping always stays on pycolmap (no GPU path "
                        "upstream). Requires --colmap-exe to point at a "
                        "CUDA build.")
    p.add_argument("--colmap-exe", default=DEFAULT_COLMAP_EXE,
                   help=f"Path to CUDA colmap.exe for --gpu mode "
                        f"(default: {DEFAULT_COLMAP_EXE})")
    p.add_argument("--gpu-index", type=int, default=0,
                   help="CUDA device index for --gpu mode (default 0). "
                        "With both a 7900 XTX and a 1070 Ti installed, "
                        "the 1070 Ti is the only CUDA device, so 0.")
    args = p.parse_args()

    dataset_dir = args.dataset_dir
    image_dir = os.path.join(dataset_dir, args.image_subdir)
    db_path = os.path.join(dataset_dir, "database.db")
    sparse_dir = os.path.join(dataset_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")

    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "SIMPLE_RADIAL"
    if args.camera_params:
        reader_opts.camera_params = args.camera_params

    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.sift.first_octave = 0  # don't upscale (saves ~4x RAM)
    extraction_opts.sift.max_num_features = 8192
    extraction_opts.num_threads = args.num_threads

    print(f"Dataset: {dataset_dir}")
    print(f"Images:  {image_dir} ({len(os.listdir(image_dir))} files)")
    print(f"Camera:  SIMPLE_RADIAL (shared)"
          + (f" primed={args.camera_params}" if args.camera_params else ""))
    print()

    t0 = time.time()
    if args.gpu:
        if not os.path.exists(args.colmap_exe):
            raise SystemExit(
                f"--gpu set but CUDA colmap.exe not found at "
                f"{args.colmap_exe}. Pass --colmap-exe <path>.")
        print(f"GPU mode: {args.colmap_exe} (device {args.gpu_index})\n")
        gpu_extract_and_match(args.colmap_exe, db_path, image_dir, args)
        print(f"  Stages 1+2 done in {time.time()-t0:.1f}s")
        sys.stdout.flush()
    else:
        print("Step 1/3: Extracting features (SIFT, first_octave=0)...")
        sys.stdout.flush()
        pycolmap.extract_features(
            database_path=db_path,
            image_path=image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_opts,
            extraction_options=extraction_opts,
            device=pycolmap.Device.cpu,
        )
        print(f"  Done in {time.time()-t0:.1f}s")
        sys.stdout.flush()

        t1 = time.time()
        print("Step 2/3: Exhaustive matching...")
        sys.stdout.flush()
        matching_opts = pycolmap.FeatureMatchingOptions()
        matching_opts.num_threads = args.num_threads
        pycolmap.match_exhaustive(
            database_path=db_path,
            matching_options=matching_opts,
            device=pycolmap.Device.cpu,
        )
        print(f"  Done in {time.time()-t1:.1f}s")
        sys.stdout.flush()

    t2 = time.time()
    print("Step 3/3: Incremental mapping...")
    sys.stdout.flush()
    pipeline_opts = pycolmap.IncrementalPipelineOptions()
    pipeline_opts.multiple_models = False
    pipeline_opts.ba_refine_principal_point = False
    pipeline_opts.ba_refine_extra_params = not args.no_refine_extra
    pipeline_opts.num_threads = args.num_threads

    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=sparse_dir,
        options=pipeline_opts,
    )
    print(f"  Done in {time.time()-t2:.1f}s")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Reconstructions: {len(reconstructions)}")

    for idx, recon in reconstructions.items():
        n_images = recon.num_reg_images()
        n_points = recon.num_points3D()
        print(f"  Model {idx}: {n_images} registered images, {n_points} 3D points")

        cam = list(recon.cameras.values())[0]
        print(f"  Camera: {cam.model_name} {cam.width}x{cam.height} params={list(cam.params)}")

if __name__ == "__main__":
    main()
