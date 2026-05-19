"""Cross-eval: Run B renders vs Run A's blurry GT images.

Evaluates the restored-model renders against the original blurry val images
to see "how close is the deblurred 3D model to the blurry reality."
"""

import sys
import os
import json
import gc
import numpy as np
import cv2
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def main():
    run_a_dir = r"E:\vksplat_output\kitchen_ab_clean\20260518_170813_kitchen"
    run_b_dir = r"E:\vksplat_output\kitchen_ab_restored\20260518_171134_kitchen"

    # Load Run A's train.json to get the blurry val image paths
    with open(os.path.join(run_a_dir, "train.json")) as f:
        train_a = json.load(f)
    blurry_val = train_a["val_images"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    psnr_fun = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fun = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_vgg_fun = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)
    lpips_alex_fun = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    def load_image(filename):
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).float().to(device) / (65535 if im.dtype == np.uint16 else 255)
        return im[None].permute(0, 3, 1, 2)

    all_metrics = []
    for i, val_entry in enumerate(blurry_val):
        blurry_path = val_entry["image_path"]
        render_path = os.path.join(run_b_dir, f"val_{i:05d}.png")

        if not os.path.exists(render_path):
            print(f"  Skipping {i}: {render_path} not found", flush=True)
            continue

        ref = load_image(blurry_path)
        im = load_image(render_path)

        with torch.no_grad():
            metrics = [
                psnr_fun(im, ref).item(),
                ssim_fun(im, ref).item(),
                lpips_vgg_fun(im, ref).item(),
                lpips_alex_fun(im, ref).item(),
            ]
        all_metrics.append(metrics)

        del ref, im
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  [{i+1}/{len(blurry_val)}] PSNR={metrics[0]:.2f} SSIM={metrics[1]:.3f} LPIPS={metrics[2]:.3f}", flush=True)

    if not all_metrics:
        print("No images evaluated")
        return

    psnr, ssim, lpips_vgg, lpips_alex = np.mean(all_metrics, axis=0)
    print(f"\n--- Run B renders vs blurry GT (Run A images) ---")
    print(f"Mean PSNR: {psnr:.2f}")
    print(f"Mean SSIM: {ssim:.3f}")
    print(f"Mean LPIPS VGG: {lpips_vgg:.3f}")
    print(f"Mean LPIPS Alex: {lpips_alex:.3f}")


if __name__ == "__main__":
    main()
