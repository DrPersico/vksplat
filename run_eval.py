"""Run eval on a VkSplat output directory in a separate process.

Loads images one at a time to minimize peak RAM usage.
Usage: python run_eval.py <output_dir>
"""

import sys
import os
import json
import gc
import numpy as np
import cv2

def main():
    output_dir = sys.argv[1]
    if not output_dir.endswith(os.sep):
        output_dir += os.sep

    train_json_path = os.path.join(output_dir, "train.json")
    with open(train_json_path) as f:
        train_data = json.load(f)

    val_images = train_data["val_images"]
    print(f"Evaluating {len(val_images)} validation images", flush=True)

    import torch
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Eval device: {device}", flush=True)

    psnr_fun = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fun = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_vgg_fun = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)
    lpips_alex_fun = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    def load_image(filename):
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).float().to(device) / (65535 if im.dtype == np.uint16 else 255)
        return im[None].permute(0, 3, 1, 2)

    obj = []
    all_metrics = []

    for i, val_entry in enumerate(val_images):
        im_train = val_entry["image_path"]
        im_render = os.path.join(output_dir, f"val_{i:05d}.png")

        if not os.path.exists(im_render):
            print(f"  Skipping {i}: {im_render} not found", flush=True)
            continue

        ref = load_image(im_train)
        im = load_image(im_render)

        with torch.no_grad():
            metrics = [
                psnr_fun(im, ref).item(),
                ssim_fun(im, ref).item(),
                lpips_vgg_fun(im, ref).item(),
                lpips_alex_fun(im, ref).item(),
            ]
        all_metrics.append(metrics)

        obj.append({
            "im_train": im_train,
            "im_render": im_render,
            "psnr": metrics[0],
            "ssim": metrics[1],
            "lpips_vgg": metrics[2],
            "lpips_alex": metrics[3],
        })

        del ref, im
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  [{i+1}/{len(val_images)}] PSNR={metrics[0]:.2f} SSIM={metrics[1]:.3f} LPIPS={metrics[2]:.3f}", flush=True)

    if not all_metrics:
        print("No val images found to evaluate", flush=True)
        return

    psnr, ssim, lpips_vgg, lpips_alex = np.mean(all_metrics, axis=0)
    print(f"\nMean PSNR: {psnr:.2f}", flush=True)
    print(f"Mean SSIM: {ssim:.3f}", flush=True)
    print(f"Mean LPIPS VGG: {lpips_vgg:.3f}", flush=True)
    print(f"Mean LPIPS Alex: {lpips_alex:.3f}", flush=True)

    result = {
        "mean": {"psnr": float(psnr), "ssim": float(ssim), "lpips_vgg": float(lpips_vgg), "lpips_alex": float(lpips_alex)},
        "images": obj,
    }
    eval_path = os.path.join(output_dir, "eval.json")
    with open(eval_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved: {eval_path}", flush=True)

if __name__ == "__main__":
    main()
