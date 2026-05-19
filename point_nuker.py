"""
PointNuker: Post-processing tool for removing unwanted Gaussian splats.

Loads a trained splat.ply and removes splats based on configurable criteria:
- Opacity threshold (near-transparent splats)
- Scale threshold (oversized floaters)
- Volume threshold (product of all 3 scales)
- Bounding box crop (AABB)
- Sphere crop (distance from center)
- Isolation filter (KNN-based outlier detection)
- Elongation filter (needle-like artifacts)

Usage:
    python point_nuker.py input.ply output.ply [filters...]
"""

import sys
import os
import argparse
import numpy as np
from time import perf_counter

sys.path.insert(0, os.path.dirname(__file__))
from ply_utils import read_ply, write_ply, print_ply_stats


def filter_opacity(data, min_opacity=0.01, max_opacity=1.0):
    """Remove splats with opacity outside [min, max] range."""
    mask = (data['opacity'] >= min_opacity) & (data['opacity'] <= max_opacity)
    return mask, f"opacity [{min_opacity:.4f}, {max_opacity:.2f}]"


def filter_scale(data, max_scale=None, min_scale=None):
    """Remove splats where any axis scale exceeds max_scale or is below min_scale."""
    mask = np.ones(data['num_splats'], dtype=bool)
    desc_parts = []
    if max_scale is not None:
        max_per_splat = data['scales'].max(axis=1)
        mask &= max_per_splat <= max_scale
        desc_parts.append(f"max_scale<={max_scale}")
    if min_scale is not None:
        min_per_splat = data['scales'].min(axis=1)
        mask &= min_per_splat >= min_scale
        desc_parts.append(f"min_scale>={min_scale}")
    return mask, f"scale ({', '.join(desc_parts)})"


def filter_volume(data, max_volume=None, min_volume=None):
    """Remove splats where volume (product of scales) exceeds threshold."""
    vol = data['scales'][:, 0] * data['scales'][:, 1] * data['scales'][:, 2]
    mask = np.ones(data['num_splats'], dtype=bool)
    desc_parts = []
    if max_volume is not None:
        mask &= vol <= max_volume
        desc_parts.append(f"<={max_volume}")
    if min_volume is not None:
        mask &= vol >= min_volume
        desc_parts.append(f">={min_volume}")
    return mask, f"volume ({', '.join(desc_parts)})"


def filter_bbox(data, bbox_min, bbox_max):
    """Keep only splats within axis-aligned bounding box."""
    xyz = data['xyz']
    mask = np.all(xyz >= bbox_min, axis=1) & np.all(xyz <= bbox_max, axis=1)
    return mask, f"bbox [{bbox_min} to {bbox_max}]"


def filter_sphere(data, center, radius):
    """Keep only splats within a sphere."""
    dists = np.linalg.norm(data['xyz'] - center, axis=1)
    mask = dists <= radius
    return mask, f"sphere (center={center}, r={radius:.3f})"


def filter_isolation(data, k=5, percentile=99.0):
    """
    Remove splats whose average K-nearest-neighbor distance exceeds
    the given percentile of all splats' KNN distances.
    Detects isolated floaters far from the main reconstruction.
    """
    from scipy.spatial import cKDTree

    t0 = perf_counter()
    tree = cKDTree(data['xyz'])
    distances, _ = tree.query(data['xyz'], k=k+1)  # +1 because first neighbor is self
    avg_knn_dist = distances[:, 1:].mean(axis=1)  # exclude self

    threshold = np.percentile(avg_knn_dist, percentile)
    mask = avg_knn_dist <= threshold
    t1 = perf_counter()

    return mask, f"isolation (k={k}, percentile={percentile}%, threshold={threshold:.5f}, took {t1-t0:.1f}s)"


def filter_elongation(data, max_ratio=50.0):
    """Remove needle-like splats where max_scale/min_scale exceeds ratio."""
    scales = data['scales']
    max_s = scales.max(axis=1)
    min_s = scales.min(axis=1).clip(1e-10)
    ratio = max_s / min_s
    mask = ratio <= max_ratio
    return mask, f"elongation (max_ratio={max_ratio})"


def filter_sh_energy(data, max_energy=None, min_dc_brightness=None):
    """
    Filter by SH coefficient properties.
    - max_energy: remove splats with excessive higher-order SH energy (noisy)
    - min_dc_brightness: remove splats that are too dark
    """
    mask = np.ones(data['num_splats'], dtype=bool)
    desc_parts = []
    
    if max_energy is not None:
        rest_sh = data['sh_coeffs'][:, 1:, :]  # higher-order coefficients
        energy = (rest_sh ** 2).sum(axis=(1, 2))
        mask &= energy <= max_energy
        desc_parts.append(f"max_energy<={max_energy}")
    
    if min_dc_brightness is not None:
        dc = data['sh_coeffs'][:, 0, :]  # DC component (N, 3)
        brightness = dc.mean(axis=1)
        mask &= brightness >= min_dc_brightness
        desc_parts.append(f"min_dc>={min_dc_brightness}")
    
    return mask, f"sh_energy ({', '.join(desc_parts)})"


def apply_mask(data, mask):
    """Apply a boolean mask to filter all arrays in the data dict."""
    return {
        'xyz': data['xyz'][mask],
        'sh_coeffs': data['sh_coeffs'][mask],
        'opacity': data['opacity'][mask],
        'scales': data['scales'][mask],
        'rotations': data['rotations'][mask],
        'num_splats': int(mask.sum()),
    }


def auto_center(data):
    """Compute the center of the scene as median position."""
    return np.median(data['xyz'], axis=0)


def auto_radius(data, percentile=99.5):
    """Compute a reasonable radius that contains most splats."""
    center = auto_center(data)
    dists = np.linalg.norm(data['xyz'] - center, axis=1)
    return np.percentile(dists, percentile)


def main():
    p = argparse.ArgumentParser(
        description="PointNuker: Remove unwanted Gaussian splats from a trained model")
    p.add_argument("input", help="Input splat.ply path")
    p.add_argument("output", help="Output filtered splat.ply path")

    # Filters
    p.add_argument("--min-opacity", type=float, default=None,
                   help="Remove splats with opacity below this (e.g. 0.01)")
    p.add_argument("--max-opacity", type=float, default=None,
                   help="Remove splats with opacity above this")
    p.add_argument("--max-scale", type=float, default=None,
                   help="Remove splats where any axis scale exceeds this")
    p.add_argument("--min-scale", type=float, default=None,
                   help="Remove splats where any axis scale is below this")
    p.add_argument("--max-volume", type=float, default=None,
                   help="Remove splats where scale_x*scale_y*scale_z exceeds this")
    p.add_argument("--bbox", type=str, default=None,
                   help="Keep only within AABB: 'x0,y0,z0,x1,y1,z1'")
    p.add_argument("--sphere-radius", type=float, default=None,
                   help="Keep only within this radius of scene center")
    p.add_argument("--sphere-center", type=str, default=None,
                   help="Center for sphere crop: 'x,y,z' (default: median)")
    p.add_argument("--isolation-percentile", type=float, default=None,
                   help="Remove splats above this KNN-distance percentile (e.g. 99)")
    p.add_argument("--isolation-k", type=int, default=5,
                   help="Number of neighbors for isolation filter")
    p.add_argument("--max-elongation", type=float, default=None,
                   help="Remove splats where max_scale/min_scale > ratio")
    p.add_argument("--max-sh-energy", type=float, default=None,
                   help="Remove splats with excessive higher-order SH energy")

    # Presets
    p.add_argument("--preset", choices=["gentle", "moderate", "aggressive"],
                   default=None,
                   help="Use a filter preset instead of manual params")

    # Options
    p.add_argument("--stats", action="store_true",
                   help="Print detailed statistics before/after filtering")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be removed without writing output")

    args = p.parse_args()

    # Apply preset defaults
    if args.preset == "gentle":
        if args.min_opacity is None: args.min_opacity = 0.005
        if args.max_scale is None: args.max_scale = 1.0
        if args.max_elongation is None: args.max_elongation = 100.0
    elif args.preset == "moderate":
        if args.min_opacity is None: args.min_opacity = 0.005
        if args.max_scale is None: args.max_scale = 0.5
        if args.max_volume is None: args.max_volume = 0.01
        if args.isolation_percentile is None: args.isolation_percentile = 99.5
        if args.max_elongation is None: args.max_elongation = 200.0
    elif args.preset == "aggressive":
        if args.min_opacity is None: args.min_opacity = 0.02
        if args.max_scale is None: args.max_scale = 0.3
        if args.max_volume is None: args.max_volume = 0.005
        if args.isolation_percentile is None: args.isolation_percentile = 98.0
        if args.max_elongation is None: args.max_elongation = 30.0

    # Load
    print(f"Loading: {args.input}")
    data = read_ply(args.input)
    original_count = data['num_splats']
    print(f"  Loaded {original_count} splats")
    if args.stats:
        print_ply_stats(data)
    print()

    # Pre-filter: remove NaN/inf positions (always applied)
    finite_mask = np.isfinite(data['xyz']).all(axis=1)
    finite_mask &= np.isfinite(data['scales']).all(axis=1)
    finite_mask &= np.isfinite(data['opacity'])
    n_invalid = (~finite_mask).sum()
    if n_invalid > 0:
        print(f"  Pre-filter: removing {n_invalid} splats with NaN/inf values")
        data = apply_mask(data, finite_mask)
        original_count = data['num_splats']
        print(f"  Remaining: {original_count} valid splats")
        print()

    # Apply filters
    filters_applied = []
    combined_mask = np.ones(original_count, dtype=bool)

    if args.min_opacity is not None or args.max_opacity is not None:
        mask, desc = filter_opacity(
            data,
            min_opacity=args.min_opacity or 0.0,
            max_opacity=args.max_opacity or 1.0
        )
        removed = (~mask).sum()
        filters_applied.append((desc, removed))
        combined_mask &= mask

    if args.max_scale is not None or args.min_scale is not None:
        mask, desc = filter_scale(data, max_scale=args.max_scale, min_scale=args.min_scale)
        removed = (~mask & combined_mask).sum()
        filters_applied.append((desc, removed))
        combined_mask &= mask

    if args.max_volume is not None:
        mask, desc = filter_volume(data, max_volume=args.max_volume)
        removed = (~mask & combined_mask).sum()
        filters_applied.append((desc, removed))
        combined_mask &= mask

    if args.bbox is not None:
        coords = [float(x) for x in args.bbox.split(',')]
        assert len(coords) == 6, "bbox must be 'x0,y0,z0,x1,y1,z1'"
        bbox_min = np.array(coords[:3])
        bbox_max = np.array(coords[3:])
        mask, desc = filter_bbox(data, bbox_min, bbox_max)
        removed = (~mask & combined_mask).sum()
        filters_applied.append((desc, removed))
        combined_mask &= mask

    if args.sphere_radius is not None:
        if args.sphere_center:
            center = np.array([float(x) for x in args.sphere_center.split(',')])
        else:
            center = auto_center(data)
        mask, desc = filter_sphere(data, center, args.sphere_radius)
        removed = (~mask & combined_mask).sum()
        filters_applied.append((desc, removed))
        combined_mask &= mask

    if args.max_elongation is not None:
        mask, desc = filter_elongation(data, max_ratio=args.max_elongation)
        removed = (~mask & combined_mask).sum()
        filters_applied.append((desc, removed))
        combined_mask &= mask

    if args.isolation_percentile is not None:
        mask, desc = filter_isolation(data, k=args.isolation_k, percentile=args.isolation_percentile)
        removed = (~mask & combined_mask).sum()
        filters_applied.append((desc, removed))
        combined_mask &= mask

    if args.max_sh_energy is not None:
        mask, desc = filter_sh_energy(data, max_energy=args.max_sh_energy)
        removed = (~mask & combined_mask).sum()
        filters_applied.append((desc, removed))
        combined_mask &= mask

    # Report
    total_removed = original_count - combined_mask.sum()
    print("Filter results:")
    for desc, removed in filters_applied:
        print(f"  {desc}: -{removed} splats")
    print()
    print(f"  Total: {original_count} -> {combined_mask.sum()} "
          f"(-{total_removed}, {100*total_removed/original_count:.1f}% removed)")

    if total_removed == 0:
        print("\nNo splats removed. Output will be identical to input.")
        if not args.dry_run:
            # Still write it for pipeline convenience
            import shutil
            shutil.copy2(args.input, args.output)
            print(f"Copied to: {args.output}")
        return

    if args.dry_run:
        print("\n[DRY RUN] No output written.")
        return

    # Apply filter and write
    filtered_data = apply_mask(data, combined_mask)
    print(f"\nWriting: {args.output}")
    write_ply(
        args.output,
        filtered_data['xyz'],
        filtered_data['sh_coeffs'],
        filtered_data['opacity'],
        filtered_data['scales'],
        filtered_data['rotations'],
    )
    print(f"  Written {filtered_data['num_splats']} splats")

    if args.stats:
        print("\nFiltered stats:")
        print_ply_stats(filtered_data)


if __name__ == "__main__":
    main()
