"""Run COLMAP SfM via pycolmap with SEQUENTIAL matching.

For sequentially-captured photo sets (e.g. timestamped frames from a phone walk-through),
sequential matching is O(N) instead of O(N²) and produces better reconstructions than
exhaustive matching for video-like capture (avoids spurious matches between visually
similar but spatially distant viewpoints).

Note: this pycolmap install does NOT have GPU SIFT support compiled in, so all stages
run on CPU regardless of the --device flag. See memory.md for related context.

Usage: python run_colmap_sequential.py <dataset_dir> [--image-subdir images_2]
                                       [--overlap 10] [--loop-detection]
Outputs to <dataset_dir>/sparse/0/
"""

import sys
import os
import time
import argparse
import pycolmap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir")
    p.add_argument("--image-subdir", default="images_2",
                   help="Subdirectory with images (default: images_2)")
    p.add_argument("--overlap", type=int, default=10,
                   help="Match each image to next N neighbors (default: 10)")
    p.add_argument("--no-quadratic-overlap", action="store_true",
                   help="Disable quadratic overlap (exp-spaced extra pairs)")
    p.add_argument("--loop-detection", action="store_true",
                   help="Enable vocab-tree loop closure (requires vocab tree path)")
    p.add_argument("--vocab-tree", default=None,
                   help="Path to vocab tree .bin file (only if --loop-detection)")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--max-features", type=int, default=8192)
    args = p.parse_args()

    if args.loop_detection and not args.vocab_tree:
        print("ERROR: --loop-detection requires --vocab-tree <path>", file=sys.stderr)
        sys.exit(1)

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

    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.sift.first_octave = 0
    extraction_opts.sift.max_num_features = args.max_features
    extraction_opts.num_threads = args.num_threads

    n_images = len(os.listdir(image_dir))
    print(f"Dataset:  {dataset_dir}")
    print(f"Images:   {image_dir} ({n_images} files)")
    print(f"Camera:   SIMPLE_RADIAL (shared)")
    print(f"Matching: sequential (overlap={args.overlap}, "
          f"quadratic={not args.no_quadratic_overlap}, "
          f"loop_detection={args.loop_detection})")
    print()

    t0 = time.time()
    print("Step 1/3: Extracting features (SIFT, first_octave=0, CPU)...")
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
    print(f"Step 2/3: Sequential matching (overlap={args.overlap})...")
    sys.stdout.flush()
    matching_opts = pycolmap.FeatureMatchingOptions()
    matching_opts.num_threads = args.num_threads

    pairing_opts = pycolmap.SequentialPairingOptions()
    pairing_opts.overlap = args.overlap
    pairing_opts.quadratic_overlap = not args.no_quadratic_overlap
    pairing_opts.loop_detection = args.loop_detection
    if args.loop_detection:
        pairing_opts.vocab_tree_path = args.vocab_tree
    pairing_opts.num_threads = args.num_threads

    pycolmap.match_sequential(
        database_path=db_path,
        matching_options=matching_opts,
        pairing_options=pairing_opts,
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
    pipeline_opts.ba_refine_extra_params = True
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
        n_reg = recon.num_reg_images()
        n_points = recon.num_points3D()
        print(f"  Model {idx}: {n_reg}/{n_images} registered images, {n_points} 3D points")
        cam = list(recon.cameras.values())[0]
        print(f"  Camera: {cam.model_name} {cam.width}x{cam.height} params={list(cam.params)}")


if __name__ == "__main__":
    main()
