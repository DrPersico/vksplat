# Capture Guide — getting a SHARP dataset for VkSplat

## Why this guide exists

Two attempts failed for the same reason:

| Attempt | Frames | Sharpness (var-of-Laplacian) | Result |
|---|---|---|---|
| 900 photos | 900 | median ~15 (sharp = 500+) | PSNR 24 but visibly blurry mush |
| 4K60 video | 29,265 | median ~15, max ~30 | identical blur — no sharp frames |

**The problem is not the camera resolution or frame rate. It is motion blur from a long per-frame exposure in a dim room.** In low light the phone keeps the shutter open longer to collect enough light; if the phone is moving at all during that window, every frame smears. 60 fps does not help, because each individual 60 fps frame still used a long exposure.

3D Gaussian Splatting can only reconstruct detail that exists in the input. Blurry in → blurry out. The fix is entirely in **how you capture**, not in training settings.

## The single most important thing: LIGHT

More light is the only way to force a short exposure, and a short exposure is the only way to stop motion blur.

- Capture in **daytime** with **all blinds/curtains open**.
- Turn on **every lamp and ceiling light** in the room.
- Add any extra lights you have (desk lamps, work lights, even a second phone's torch held by someone else) pointed at dim corners.
- Do **not** shoot toward the bright window with a dark interior — the camera exposes for the dark room and lengthens the shutter. Keep the window behind or to the side of you, or wait for overcast daylight that balances indoor/outdoor brightness.

If you do nothing else, do this. A bright room with a phone on auto will already be far sharper than the last two attempts.

## Recommended method: stop-and-shoot PHOTOS (most reliable)

This eliminates motion blur by guaranteeing the phone is still during each shot.

1. Stand still. Brace the phone with **both hands**, elbows against your body. Better: a small tripod or gimbal.
2. Take a photo. Wait ~1 second for the phone to fully settle.
3. Take **2–3 steps**, stop, let the phone settle again (~1 s), take the next photo.
4. Never press the shutter while walking or panning.
5. Walk a slow loop around the room, then a second loop at a different height (e.g., lower) for vertical coverage. Overlap each shot ~70% with the previous one.
6. Target **150–300 photos**. More is not better if they are redundant; coverage of all viewpoints matters more than count (see memory.md: 249 sharp-ish photos beat 85, but quality plateaus).
7. Avoid pointing straight at mirrors, TVs/screens, or the bare window.

### If your phone has Pro / Manual mode — use it

- Shutter speed: **1/125 s or faster** (1/250 s if the room is bright enough).
- ISO: let it rise to compensate (ISO 800–3200 noise is far less damaging than motion blur — splatting averages noise across views, but cannot remove blur).
- Focus: manual or tap-to-focus on a mid-distance object; lock it so it does not hunt between shots.
- White balance: lock it so colors are consistent across the dataset.

### If only auto mode

- Tap to focus on a well-lit mid-distance object, then use **AE/AF lock** (long-press on most phones) so exposure does not jump when you frame the window.
- Disable any "night mode" / long-exposure mode — it multiplies the blur.

## Alternative: video — ONLY in a bright room

Video is acceptable *if and only if* the room is bright enough that the phone runs a short exposure. The selection script will pick the sharpest frames automatically.

1. Light the room as above (mandatory).
2. Lock exposure/focus before recording (AE/AF lock).
3. Record at the highest resolution your phone offers. 30 fps is fine; 60 fps gives the selector more candidates but does not improve individual-frame sharpness.
4. Walk **slowly and smoothly**. Pause for ~2 s every few steps — those pauses produce the sharpest frames.
5. Do a slow full loop, then a second loop at a different height.
6. 3–6 minutes of footage is plenty.

## Verify BEFORE building the whole dataset (do not skip this)

After capturing ~10–20 test shots (or ~20 s of test video), run the sharpness check. This takes under a minute and saves ~90 minutes of COLMAP on bad data.

```powershell
# photos:
python E:\vksplat\select_sharp_frames.py --photos "C:\path\to\test\shots" --check

# video:
python E:\vksplat\select_sharp_frames.py --video "C:\path\to\test.mp4" --fps 4 --check
```

Read the printed median variance-of-Laplacian:

- **median > 300** — excellent, proceed with full capture.
- **median 150–300** — usable; more light would still help.
- **median < 150** — still too blurry. **Stop. Add light, slow down, retry.** Do not capture the full dataset yet.

For reference, the failed attempts scored **~15**.

## Once you have a sharp dataset

```powershell
# Photos:
python E:\vksplat\select_sharp_frames.py --photos "C:\path\to\shots" --out-root E:\vksplat_data\sharp --target-count 300

# Video:
python E:\vksplat\select_sharp_frames.py --video "C:\path\to\clip.mp4" --out-root E:\vksplat_data\sharp --target-count 400

# Then the existing, already-validated pipeline:
python E:\vksplat\run_colmap_sequential.py E:\vksplat_data\sharp --image-subdir images_2
python E:\vksplat\train_livingroom.py --dataset-dir E:\vksplat_data\sharp --image-dir images_4 --strategy mcmc --cap-max 500000 --steps 5000 --skip-eval   # smoke
python E:\vksplat\train_livingroom.py --dataset-dir E:\vksplat_data\sharp --image-dir images_4 --strategy mcmc --cap-max 1000000 --steps 30000 --tag sharp_mcmc1m   # full
```

Do **not** raise `--cap-max` above 1,000,000 — 2M produced degenerate fog and 3M hung the AMD driver (see memory.md). If the source is genuinely sharp, instead try `--image-dir images_2` on a ~450-frame subset; higher training resolution only helps when the source actually has detail to resolve.

## When you can't reshoot: GPU deblur salvage

Everything above assumes you can still go back and capture properly. Sometimes you can't — the location is far away, no longer accessible, or the only photos you have are the imperfect ones. `restore_frames.py` deblurs an already-staged dataset with a pretrained NAFNet network so a motion-blurred capture stays usable. It is a salvage tool, **not** a substitute for a sharp capture — a real deblur recovers some detail but cannot invent what the long exposure destroyed. Capture sharp when you can.

### The one rule: training tier ONLY, never the COLMAP tier

Run restoration **only** on the training tier (`images_4`). Never on `images_2`.

COLMAP matches SIFT features *across* views. Deblur output is not view-consistent — each frame is restored independently and the network can synthesise slightly different texture per view. Feeding that to COLMAP corrupts feature matching and poses (the pipeline is extremely pose-sensitive; see memory.md issue #1). Poses must come from the clean, faithful `images_2` pixels. By the time training reads `images_4`, poses are already locked, so detail recovery there helps the photometric loss without touching geometry. The script refuses to run if pointed at an `images_2` folder.

### Workflow

```powershell
# 1. Stage + COLMAP exactly as normal, on the ORIGINAL (blurry) frames.
python E:\vksplat\select_sharp_frames.py --photos "C:\blurry\shots" --out-root E:\vksplat_data\salvage --target-count 300
python E:\vksplat\run_colmap_sequential.py E:\vksplat_data\salvage --image-subdir images_2

# 2. Dry-run: probe the GPU + see the sharpness gain on a sample (writes nothing).
python E:\vksplat\restore_frames.py E:\vksplat_data\salvage\images_4 --dry-run

# 3. Restore the training tier -> sibling images_4_restored\ (originals untouched).
python E:\vksplat\restore_frames.py E:\vksplat_data\salvage\images_4

# 4. Train against the restored tier — same pipeline, just a different --image-dir.
python E:\vksplat\train_livingroom.py --dataset-dir E:\vksplat_data\salvage --image-dir images_4_restored --strategy mcmc --cap-max 1000000 --steps 30000 --tag salvage_dl
#    A/B it against the unrestored baseline (--image-dir images_4) and compare eval.json.
```

### Which Python to run it with (IMPORTANT)

NAFNet's GPU path needs a **CUDA** torch. The default `python` on this machine has the **AMD ROCm** torch (for the RX 7900 XTX / VkSplat) — with it, the GPU probe fails and the script falls back to slow CPU/classical. To run NAFNet on the **GTX 1070 Ti**, use the NVIDIA CUDA env instead:

```powershell
# WRONG env (ROCm torch -> falls back to CPU/classical):
python E:\vksplat\restore_frames.py ...

# RIGHT env for 1070 Ti GPU deblur (torch 2.5.1+cu121, sees the 1070 Ti):
E:\nvidia-gsplat\python312\python.exe E:\vksplat\restore_frames.py E:\vksplat_data\salvage\images_4 --method dl
```

The 1070 Ti (Pascal sm_61) is fine here even though it's a dead-end for gsplat: NAFNet is **inference-only** (no backward pass), so it doesn't hit the Volta-only backward kernels that block gsplat. GPU deblur is ~100× faster than `--dl-cpu` (full 900-frame restore: minutes vs ~12 h). `restore_frames.py` adds its own dir to `sys.path`, so it runs from any cwd/env.

### GPU, fallback, and tuning

- **Primary path (`--method dl`, default):** NAFNet-GoPro-width64 on the GPU, inference-only. Weights live at `E:\vksplat_tools\models\NAFNet-GoPro-width64.pth`.
- **GPU probe + auto-fallback:** the script runs a real tensor op on the GPU before trusting it (a ROCm torch reports a build but fails device enumeration; a CUDA torch in the NVIDIA env passes and uses the 1070 Ti). If the GPU isn't usable it prints a clear warning and **falls back to classical `unsharp`** so the run still produces a restored tier — it never hard-blocks.
- **`--dl-cpu`:** run NAFNet on CPU. Correct output, but slow (~0.02 img/s — minutes per frame); only practical for small sets or validation.
- **`--cpu` / `--method unsharp|wiener|denoise-sharpen`:** classical CPU restoration. `unsharp` is the safe default fallback; `denoise-sharpen` suits soft-and-grainy dim-room frames.
- **`--dl-strength` (default 0.7):** blends NAFNet output with the original. Lower it (e.g. 0.5) if you see any over-sharpened or synthesised texture; raise toward 1.0 for heavier blur.
- A per-frame guardrail keeps the original for any frame the method makes *less* sharp or *newly* blows out (it tolerates bright windows/lamps that were already clipped in the source).

Decision: if the dry-run median sharpness delta is solidly positive and a visual spot-check of a few `images_4_restored\` frames shows crisper edges with no ringing/hallucination, train against it and let the A/B `eval.json` (PSNR/SSIM/LPIPS-Alex vs the unrestored baseline) be the final judge.
