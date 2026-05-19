"""Resume COLMAP incremental mapping from an already-populated database.

When run_colmap_sequential.py completes feature extraction + matching but the
process dies during/after incremental mapping (no sparse output persisted), the
database.db still holds all keypoints/descriptors/matches/two_view_geometries.
Re-running the full pipeline would waste hours re-extracting + re-matching.

This script runs ONLY pycolmap.incremental_mapping against the existing DB and
writes to a fresh output dir (never clobbers a prior sparse/0).

Usage: python colmap_map_only.py <dataset_dir> --image-subdir images_2_unsharp
                                  [--out sparse_unsharp]
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
                   help="Image dir whose features are already in database.db")
    p.add_argument("--out", default=None,
                   help="Output subdir name (default: sparse_<imagesubdir>)")
    p.add_argument("--camera-params", default=None,
                   help="Prime intrinsics 'f,cx,cy,k' (SIMPLE_RADIAL). "
                        "Overwrites the camera row in the DB before mapping. "
                        "Fixes the degenerate-k divergence (memory Issue #10).")
    p.add_argument("--no-refine-extra", action="store_true",
                   help="Disable ba_refine_extra_params (keep primed k fixed)")
    p.add_argument("--no-refine-focal", action="store_true",
                   help="Disable ba_refine_focal_length (keep primed f fixed)")
    p.add_argument("--num-threads", type=int, default=4)
    args = p.parse_args()

    dataset_dir = args.dataset_dir
    image_dir = os.path.join(dataset_dir, args.image_subdir)
    src_db = os.path.join(dataset_dir, "database.db")
    out_name = args.out or f"sparse_{args.image_subdir}"
    out_dir = os.path.join(dataset_dir, out_name)

    if not os.path.exists(src_db):
        print(f"ERROR: no database at {src_db}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    # When priming, work on a per-run COPY so the populated source DB
    # (expensive extraction+matching) stays pristine for other priors.
    if args.camera_params:
        import shutil
        db_path = os.path.join(out_dir, "database.db")
        if not os.path.exists(db_path):
            print(f"Copying DB -> {db_path} (preserve source)")
            sys.stdout.flush()
            shutil.copy2(src_db, db_path)
    else:
        db_path = src_db

    n_images = len(os.listdir(image_dir)) if os.path.isdir(image_dir) else -1
    print(f"Dataset:  {dataset_dir}")
    print(f"DB:       {db_path}")
    print(f"Images:   {image_dir} ({n_images} files)")
    print(f"Output:   {out_dir}")
    print("Step: Incremental mapping ONLY (extraction + matching reused from DB)")

    if args.camera_params:
        # pycolmap.Database is abstract in this build; edit the SQLite
        # cameras table directly. COLMAP schema: model INTEGER (SIMPLE_RADIAL
        # = 2), params BLOB (little-endian float64 array), prior_focal_length.
        import sqlite3
        import struct
        vals = [float(x) for x in args.camera_params.split(",")]
        SIMPLE_RADIAL = 2
        blob = struct.pack("<{}d".format(len(vals)), *vals)
        con = sqlite3.connect(db_path)
        rows = con.execute(
            "SELECT camera_id, model, params FROM cameras").fetchall()
        for cid, model, old_blob in rows:
            n_old = len(old_blob) // 8
            old = list(struct.unpack("<{}d".format(n_old), old_blob))
            con.execute(
                "UPDATE cameras SET model=?, params=?, "
                "prior_focal_length=1 WHERE camera_id=?",
                (SIMPLE_RADIAL, blob, cid))
            print(f"  Primed camera {cid}: model {model} {old} "
                  f"-> model {SIMPLE_RADIAL} {vals}")
        con.commit()
        con.close()
    print(f"BA: refine_focal={not args.no_refine_focal} "
          f"refine_extra={not args.no_refine_extra}")
    sys.stdout.flush()

    t0 = time.time()
    pipeline_opts = pycolmap.IncrementalPipelineOptions()
    pipeline_opts.multiple_models = False
    pipeline_opts.ba_refine_principal_point = False
    pipeline_opts.ba_refine_extra_params = not args.no_refine_extra
    pipeline_opts.ba_refine_focal_length = not args.no_refine_focal
    pipeline_opts.num_threads = args.num_threads

    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=out_dir,
        options=pipeline_opts,
    )
    dt = time.time() - t0
    print(f"\nMapping done in {dt:.1f}s")
    print(f"Reconstructions: {len(reconstructions)}")
    for idx, recon in reconstructions.items():
        n_reg = recon.num_reg_images()
        n_pts = recon.num_points3D()
        pct = (100.0 * n_reg / n_images) if n_images > 0 else float("nan")
        print(f"  Model {idx}: {n_reg}/{n_images} ({pct:.1f}%) registered, "
              f"{n_pts} 3D points")
        cam = list(recon.cameras.values())[0]
        print(f"  Camera: {cam.model_name} {cam.width}x{cam.height} "
              f"params={list(cam.params)}")


if __name__ == "__main__":
    main()
