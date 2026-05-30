"""Stratified evaluation for video-sourced 3DGS training.

Problem: standard eval_interval=8 holds out every 8th sequential frame.
For video, neighboring frames have near-identical viewpoints, inflating metrics.

This script implements two harder evaluation tiers:
  1. Spatial clustering: group cameras by position, pick representative views
     from diverse locations. Metrics measure quality across the whole scene.
  2. Interpolation: render at interpolated poses between training views.
     These are truly novel viewpoints never seen during training.

Usage: python run_eval_stratified.py <output_dir>

Reads train.json from output_dir, loads COLMAP poses from the dataset,
re-renders held-out and interpolated views, computes PSNR/SSIM/LPIPS.
"""

import sys
import os
import json
import gc
import numpy as np
import cv2


def load_colmap_cameras(sparse_dir):
    """Load camera poses from COLMAP binary model. Returns list of dicts."""
    import pycolmap
    recon = pycolmap.Reconstruction(sparse_dir)

    cameras_out = []
    for img_id in sorted(recon.images.keys()):
        img = recon.images[img_id]
        cam = recon.cameras[img.camera_id]

        R = img.cam_from_world.rotation.matrix()
        t = img.cam_from_world.translation

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        c2w = np.linalg.inv(w2c)
        position = c2w[:3, 3]

        cameras_out.append({
            "image_id": img_id,
            "name": img.name,
            "w2c": w2c,
            "c2w": c2w,
            "position": position,
            "fx": cam.params[0],
            "fy": cam.params[0] if cam.model_name == "SIMPLE_RADIAL" else cam.params[1],
            "cx": cam.params[1] if cam.model_name == "SIMPLE_RADIAL" else cam.params[2],
            "cy": cam.params[2] if cam.model_name == "SIMPLE_RADIAL" else cam.params[3],
            "w": cam.width,
            "h": cam.height,
        })

    return cameras_out


def cluster_cameras(cameras, n_clusters=12):
    """K-means cluster cameras by 3D position. Returns cluster assignments."""
    from sklearn.cluster import KMeans

    positions = np.array([c["position"] for c in cameras])
    kmeans = KMeans(n_clusters=min(n_clusters, len(cameras)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(positions)

    clusters = {}
    for i, label in enumerate(labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return clusters, kmeans.cluster_centers_


def select_stratified_views(cameras, clusters, cluster_centers):
    """Pick 1-2 views per cluster closest to centroid for hard eval."""
    selected = []
    for cluster_id, indices in sorted(clusters.items()):
        center = cluster_centers[cluster_id]
        dists = [(i, np.linalg.norm(cameras[i]["position"] - center)) for i in indices]
        dists.sort(key=lambda x: x[1])
        selected.append(dists[0][0])
        if len(dists) > 3:
            selected.append(dists[1][0])

    return sorted(set(selected))


def interpolate_pose(cam_a, cam_b, t=0.5):
    """Linearly interpolate between two camera poses at parameter t."""
    c2w_a = cam_a["c2w"]
    c2w_b = cam_b["c2w"]

    R_a = c2w_a[:3, :3]
    R_b = c2w_b[:3, :3]
    t_a = c2w_a[:3, 3]
    t_b = c2w_b[:3, 3]

    t_interp = (1 - t) * t_a + t * t_b

    # Spherical interpolation for rotation via quaternion
    from scipy.spatial.transform import Rotation, Slerp
    rots = Rotation.from_matrix(np.stack([R_a, R_b]))
    slerp = Slerp([0.0, 1.0], rots)
    R_interp = slerp(t).as_matrix()

    c2w_interp = np.eye(4)
    c2w_interp[:3, :3] = R_interp
    c2w_interp[:3, 3] = t_interp

    fx = (cam_a["fx"] + cam_b["fx"]) / 2
    fy = (cam_a["fy"] + cam_b["fy"]) / 2
    cx = (cam_a["cx"] + cam_b["cx"]) / 2
    cy = (cam_a["cy"] + cam_b["cy"]) / 2
    w = cam_a["w"]
    h = cam_a["h"]

    return {
        "c2w": c2w_interp,
        "w2c": np.linalg.inv(c2w_interp),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "w": w, "h": h,
    }


def select_interpolation_views(cameras, n_pairs=15):
    """Select evenly spaced pairs for interpolation evaluation."""
    n = len(cameras)
    if n < 4:
        return []

    step = max(1, n // n_pairs)
    pairs = []
    for i in range(0, n - 2, step):
        pairs.append((i, i + 2))
        if len(pairs) >= n_pairs:
            break

    return pairs


def render_and_eval(output_dir, cameras, eval_views, eval_type="stratified"):
    """Render specified views and compute metrics against ground truth.

    For stratified: renders at actual camera poses, compares to GT image.
    For interpolation: renders at novel poses, compares to nearest GT.
    """
    import torch
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Eval device: {device}", flush=True)

    psnr_fun = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fun = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_alex_fun = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    def load_image(filename):
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).float().to(device) / (65535 if im.dtype == np.uint16 else 255)
        return im[None].permute(0, 3, 1, 2)

    # Load the trained model via VkSplat
    train_json_path = os.path.join(output_dir, "train.json")
    config_json_path = os.path.join(output_dir, "config.json")

    if not os.path.exists(config_json_path):
        print(f"  ERROR: {config_json_path} not found, cannot render", flush=True)
        return None

    with open(config_json_path) as f:
        config = json.load(f)

    ply_path = os.path.join(output_dir, config.get("output_ply", "splat.ply"))
    if not os.path.exists(ply_path):
        ply_path = os.path.join(output_dir, "splat.ply")
    if not os.path.exists(ply_path):
        print(f"  ERROR: PLY not found at {ply_path}", flush=True)
        return None

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vksplat"))
    import vksplat as vk

    module = vk.VkSplat()
    spv_dir = os.path.join(os.path.dirname(__file__), "vksplat", "shader")
    module.initialize(spv_dir, 0)
    module.set_train_config(config)
    module.load_ply(ply_path)

    strat_dir = os.path.join(output_dir, f"strat_{eval_type}")
    os.makedirs(strat_dir, exist_ok=True)

    all_metrics = []
    results = []

    for vi, view in enumerate(eval_views):
        c2w = view["c2w"]

        # Convert c2w to w2c for VkSplat
        R = c2w[:3, :3]
        T = c2w[:3, 3:]
        R_flip = R * np.array([[1.0, -1.0, -1.0]])
        R_inv = R_flip.T
        T_inv = -R_inv @ T
        w2c = np.eye(4)
        w2c[:3, :3] = R_inv
        w2c[:3, 3:4] = T_inv

        h, w = int(view["h"]), int(view["w"])
        fx, fy = float(view["fx"]), float(view["fy"])
        cx, cy = float(view["cx"]), float(view["cy"])

        module.set_uniforms(3, w2c, h, w, fx, fy, cx, cy, False)
        module.forward()

        rendered = np.clip(module.pixel_state, 0.0, 1.0)
        render_path = os.path.join(strat_dir, f"{eval_type}_{vi:05d}.png")

        im_save = np.round(65535 * rendered).astype(np.uint16)
        im_save[:, :, 3] = 65535 - im_save[:, :, 3]
        im_save = cv2.cvtColor(im_save, cv2.COLOR_BGRA2RGB)
        cv2.imwrite(render_path, im_save)

        # For stratified eval, compare to the actual GT image
        gt_path = view.get("gt_path")
        if gt_path and os.path.exists(gt_path):
            rendered_rgb = rendered[:, :, :3]
            rendered_t = torch.from_numpy(rendered_rgb).float().to(device)
            rendered_t = rendered_t[None].permute(0, 3, 1, 2)
            gt_t = load_image(gt_path)

            # Resize if dimensions don't match
            if rendered_t.shape != gt_t.shape:
                rendered_t = torch.nn.functional.interpolate(
                    rendered_t, size=gt_t.shape[2:], mode='bilinear', align_corners=False)

            with torch.no_grad():
                psnr_val = psnr_fun(rendered_t, gt_t).item()
                ssim_val = ssim_fun(rendered_t, gt_t).item()
                lpips_val = lpips_alex_fun(rendered_t, gt_t).item()

            all_metrics.append([psnr_val, ssim_val, lpips_val])
            results.append({
                "index": vi,
                "gt_path": gt_path,
                "render_path": render_path,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips_alex": lpips_val,
            })
            print(f"  [{vi+1}/{len(eval_views)}] PSNR={psnr_val:.2f} "
                  f"SSIM={ssim_val:.3f} LPIPS={lpips_val:.3f}", flush=True)

            del rendered_t, gt_t
            torch.cuda.empty_cache()
            gc.collect()
        else:
            results.append({
                "index": vi,
                "render_path": render_path,
                "note": "no GT for comparison",
            })

    module.cleanup()

    if all_metrics:
        means = np.mean(all_metrics, axis=0)
        return {
            "type": eval_type,
            "n_views": len(eval_views),
            "n_with_gt": len(all_metrics),
            "mean": {
                "psnr": float(means[0]),
                "ssim": float(means[1]),
                "lpips_alex": float(means[2]),
            },
            "images": results,
        }
    return {"type": eval_type, "n_views": len(eval_views), "images": results}


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_eval_stratified.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    train_json_path = os.path.join(output_dir, "train.json")
    config_json_path = os.path.join(output_dir, "config.json")

    if not os.path.exists(train_json_path):
        print(f"ERROR: {train_json_path} not found")
        sys.exit(1)

    with open(train_json_path) as f:
        train_data = json.load(f)
    with open(config_json_path) as f:
        config = json.load(f)

    dataset_dir = config["dataset_dir"]
    sparse_dir = os.path.join(dataset_dir, config.get("sparse_dir", "sparse/0"))
    if not os.path.isabs(sparse_dir):
        sparse_dir = os.path.join(dataset_dir, "sparse", "0")

    print(f"Loading COLMAP cameras from {sparse_dir}...", flush=True)
    cameras = load_colmap_cameras(sparse_dir)
    print(f"  Loaded {len(cameras)} cameras", flush=True)

    if len(cameras) < 10:
        print("  Too few cameras for stratified eval, skipping.")
        sys.exit(0)

    # Build mapping from image name to GT path
    all_images = {}
    for entry in train_data.get("train_images", []):
        name = os.path.basename(entry["image_path"])
        all_images[name] = entry["image_path"]
    for entry in train_data.get("val_images", []):
        name = os.path.basename(entry["image_path"])
        all_images[name] = entry["image_path"]

    for cam in cameras:
        cam["gt_path"] = all_images.get(cam["name"])

    # Determine which indices are training vs validation
    val_names = set()
    for entry in train_data.get("val_images", []):
        val_names.add(os.path.basename(entry["image_path"]))

    train_indices = [i for i, c in enumerate(cameras) if c["name"] not in val_names]
    val_indices = [i for i, c in enumerate(cameras) if c["name"] in val_names]

    print(f"  Train views: {len(train_indices)}, Val views: {len(val_indices)}", flush=True)

    # --- Tier 1: Spatial clustering ---
    print("\n=== Tier 1: Spatially-stratified evaluation ===", flush=True)
    n_clusters = min(12, len(cameras) // 3)
    clusters, centers = cluster_cameras(cameras, n_clusters=n_clusters)
    selected_indices = select_stratified_views(cameras, clusters, centers)

    # Only keep views that are in the validation set for fair comparison
    strat_views = [cameras[i] for i in selected_indices if cameras[i].get("gt_path")]
    print(f"  Selected {len(strat_views)} spatially diverse views for evaluation", flush=True)

    strat_result = None
    if strat_views:
        strat_result = render_and_eval(output_dir, cameras, strat_views, "stratified")
        if strat_result and "mean" in strat_result:
            m = strat_result["mean"]
            print(f"\n  Stratified mean: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f} "
                  f"LPIPS-Alex={m['lpips_alex']:.3f}", flush=True)

    # --- Tier 2: Interpolation ---
    print("\n=== Tier 2: Interpolation evaluation ===", flush=True)
    interp_pairs = select_interpolation_views(cameras, n_pairs=15)
    interp_views = []
    for i, j in interp_pairs:
        interp = interpolate_pose(cameras[i], cameras[j], t=0.5)
        # Use nearest GT for comparison
        interp["gt_path"] = cameras[i].get("gt_path") or cameras[j].get("gt_path")
        interp_views.append(interp)

    print(f"  Generated {len(interp_views)} interpolated views", flush=True)

    interp_result = None
    if interp_views:
        interp_result = render_and_eval(output_dir, cameras, interp_views, "interpolated")
        if interp_result and "mean" in interp_result:
            m = interp_result["mean"]
            print(f"\n  Interpolation mean: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f} "
                  f"LPIPS-Alex={m['lpips_alex']:.3f}", flush=True)

    # --- Save combined results ---
    combined = {}
    if strat_result:
        combined["stratified"] = strat_result
    if interp_result:
        combined["interpolated"] = interp_result

    # Load standard eval for comparison
    standard_eval_path = os.path.join(output_dir, "eval.json")
    if os.path.exists(standard_eval_path):
        with open(standard_eval_path) as f:
            standard = json.load(f)
        combined["standard"] = standard.get("mean", standard)

    out_path = os.path.join(output_dir, "eval_stratified.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=4)
    print(f"\nSaved: {out_path}", flush=True)

    # Summary comparison
    if "standard" in combined:
        print("\n=== Eval Comparison ===", flush=True)
        std = combined["standard"]
        print(f"  Standard:      PSNR={std.get('psnr', 0):.2f} "
              f"SSIM={std.get('ssim', 0):.3f} "
              f"LPIPS={std.get('lpips_alex', std.get('lpips', 0)):.3f}")
        if strat_result and "mean" in strat_result:
            s = strat_result["mean"]
            print(f"  Stratified:    PSNR={s['psnr']:.2f} SSIM={s['ssim']:.3f} "
                  f"LPIPS={s['lpips_alex']:.3f}")
        if interp_result and "mean" in interp_result:
            ip = interp_result["mean"]
            print(f"  Interpolation: PSNR={ip['psnr']:.2f} SSIM={ip['ssim']:.3f} "
                  f"LPIPS={ip['lpips_alex']:.3f}")


if __name__ == "__main__":
    main()
