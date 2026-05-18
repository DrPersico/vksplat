# VkSplat — Scene Reconstruction Pipeline

[![Website](https://img.shields.io/website?url=https://harry7557558.github.io/vksplat/&logo=github)](https://harry7557558.github.io/vksplat/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.00219-b31b1b.svg)](https://arxiv.org/abs/2605.00219)
![License](https://img.shields.io/github/license/harry7557558/vksplat)

This repository wraps the [VkSplat](https://harry7557558.github.io/vksplat/) Vulkan
3D Gaussian Splatting trainer with an end-to-end **photo → reconstruction**
pipeline: capture-quality checking, dataset staging, COLMAP structure-from-motion,
training, and evaluation.

> **TL;DR — train a scene from a folder of photos:**
> ```powershell
> # 1. (optional) check the photos are sharp enough
> python select_sharp_frames.py --photos "E:\Downloads\my room" --check
> # 2. stage into tiered resolutions
> python prepare_dataset.py "E:\Downloads\my room" E:\vksplat_data\myroom
> # 3. structure-from-motion (GPU COLMAP)
> python run_colmap.py E:\vksplat_data\myroom --image-subdir images_2 --gpu --camera-params 1343.6,1000,462,0.051
> # 4. train with the proven best config
> python train_livingroom.py --dataset-dir E:\vksplat_data\myroom --image-dir images_4 --strategy mcmc --cap-max 1000000 --steps 100000 --ssim-lambda 0.4 --tag myroom_best
> # 5. (optional) re-run evaluation on the saved model
> python run_eval.py E:\vksplat_output\myroom_myroom_best\<timestamp>_myroom
> ```

---

## The single most important factor: capture quality

Every quality problem in this project traced back to **motion-blurred source
photos**, not training settings. Gaussian Splatting fits splat colors directly
to pixels — blurry pixels produce a blurry ("foggy") model no matter what
config you use. COLMAP will still register blurry photos fine, so a good
registration count does **not** mean a usable dataset.

**Capture rule:** plain default phone camera, **stop-and-shoot** (stop walking,
let the phone settle ~1 s, take the shot, repeat), in a **well-lit** room.
This beat every camera gimmick tried (locked shutter, multifocus, RAW, 60 fps
video). See [`CAPTURE_GUIDE.md`](CAPTURE_GUIDE.md) for the full method.

**Sharpness gate (do this first):**
```powershell
python select_sharp_frames.py --photos "E:\Downloads\my capture" --check
```
Variance-of-Laplacian: **>300 = sharp**, 150–300 = usable, **<150 = reshoot**.
Good datasets in this project scored median ~340–390. Below ~150, do not
proceed — the reconstruction will be foggy.

---

## Pipeline

### 1. (Optional) Sharpest-frame selection — `select_sharp_frames.py`

Selects the sharpest frames from a **video** or **photo folder** with even
temporal coverage, and stages them into the tiered layout.

```powershell
# from a video
python select_sharp_frames.py --video "E:\Downloads\walk.mp4" --out-root E:\vksplat_data\myroom --target-count 400 --fps 4

# from a photo folder (just scores + selects best)
python select_sharp_frames.py --photos "E:\Downloads\my room" --out-root E:\vksplat_data\myroom --target-count 400

# scoring only, no staging (the sharpness gate)
python select_sharp_frames.py --photos "E:\Downloads\my room" --check
```
Key flags: `--target-count` (default 400), `--fps` (video sample rate),
`--start`/`--duration` (video trim), `--width-2`/`--width-4` (tier widths).

### 2. Dataset staging — `prepare_dataset.py`

Applies EXIF orientation, strips metadata, and writes resolution tiers
(`images_2`, `images_4`) used by COLMAP and training.

```powershell
python prepare_dataset.py "E:\Downloads\my room" E:\vksplat_data\myroom
```
- `--width-2` (default 2040) / `--width-4` (default 1020) — tier widths.
  These names are a **convention**, not a literal fraction of the source.
  For very large (8K) sources use `--width-2 2000 --width-4 1000`.
- `--skip-fullres` — don't also copy a full-resolution tier (saves disk).
- `--max N` — cap number of images. `--quality` — JPEG quality (default 95).

### 3. Structure-from-motion — `run_colmap.py`

COLMAP feature extraction + matching + sparse mapping → `sparse/0/`.
**Run on `images_2`** (full-res COLMAP exhausts RAM on this hardware).

```powershell
python run_colmap.py E:\vksplat_data\myroom --image-subdir images_2 --gpu --camera-params 1343.6,1000,462,0.051
```
- `--gpu` — use the CUDA `colmap.exe` (stages 1+2 GPU-accelerated; ~20 min
  vs ~8 h CPU on 1300 images). Stage 3 mapping is CPU-only regardless.
- `--camera-params "f,cx,cy,k"` — **prime the intrinsics.** COLMAP otherwise
  often estimates a degenerate focal length on these phone photos. The Samsung
  2000×924 values are `1343.6,1000,462,0.051`. Pair with `--no-refine-extra`
  to keep them fixed.
- `--num-threads` (default 12 = this box's logical cores; drop to ~4 only for
  huge 4000px+ source images that load a full image per extraction thread).
- `--colmap-exe` / `--gpu-index` — override the CUDA COLMAP path / device.

`run_colmap_sequential.py` is a sequential-matching variant — faster on
strictly time-ordered captures, but **less robust**; exhaustive (the default
above) is preferred for general use.

### 4. Training — `train_livingroom.py`

Despite the name, this is the **general training entry point** (takes
`--dataset-dir`). It loads the COLMAP model, trains, saves `splat.ply`, and
runs evaluation.

```powershell
python train_livingroom.py --dataset-dir E:\vksplat_data\myroom --image-dir images_4 --strategy mcmc --cap-max 1000000 --steps 100000 --ssim-lambda 0.4 --tag myroom_best
```

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset-dir` | `E:\vksplat_data\livingroom` | Dataset root (must contain the image tier + `sparse/`) |
| `--image-dir` | `images_2` | Training resolution tier. **`images_4` is the proven sweet spot** |
| `--strategy` | `default` | Use **`mcmc`** — far better quality than `default` (which over-densifies) |
| `--cap-max` | `2000000` | MCMC splat budget. **Use `1000000`.** Proven optimum: 500k worse, 2M/3M fog |
| `--steps` | `30000` | **`100000`** — proven best (beat 50k by ~5% LPIPS). More steps help *because* the LR schedule freezes at 30k (see `--max-steps`) |
| `--max-steps` | `None` (→30000) | LR-decay denominator. **Leave unset.** Setting it = `--steps` (stretching the schedule) causes **fog** — the 30k clamp is load-bearing |
| `--ssim-lambda` | `0.2` | **`0.4`** = best perceptual/LPIPS quality (may slightly lower PSNR) |
| `--skip-eval` | off | Skip LPIPS/SSIM eval (saves host RAM; use for quick smoke tests) |
| `--output-base` | `E:\vksplat_output` | Output parent dir |
| `--opacity-reg` / `--scale-reg` / `--noise-lr` | MCMC defaults | Fog-fighting knobs; tested, did **not** rescue high-res fog |

**Proven best config (use this):**
`--image-dir images_4 --strategy mcmc --cap-max 1000000 --steps 100000 --ssim-lambda 0.4`
(do **not** pass `--max-steps` — leaving it at the 30000 default is what makes
the long run work; matching it to `--steps` fogs.)

Output goes to `E:\vksplat_output\<datasetname>_<tag>\<timestamp>_<name>\`
containing `splat.ply`, `eval.json`, `train.json`, and `val_*.png` renders.

Other `train_*.py` scripts (`train_kitchen`, `train_photos`,
`train_combined`, `train_frames`) are thin per-dataset presets around the same
machinery; `train_livingroom.py` with `--dataset-dir` supersedes them.

### 5. Evaluation — `run_eval.py`

Re-computes PSNR/SSIM/LPIPS on a finished run (loads images one at a time to
keep peak host RAM low — training auto-evals, this is for re-running).

```powershell
python run_eval.py E:\vksplat_output\myroom_myroom_best\<timestamp>_myroom
```

---

## Hard-won lessons (do not relearn these)

- **Judge quality by LPIPS + the actual rendered image, never by PSNR alone
  or by the densification log.** Blurry-vs-blurry inflates PSNR; a high PSNR
  with a soft render is worse than a lower PSNR with a sharp one. The
  `relocate / 0 add` densification line is **ambiguous** and does *not* by
  itself indicate fog — only the eval metrics + render do.
- **Fog signature** (degenerate run): PSNR ~9–14, LPIPS-Alex ≳0.84, render is
  a uniform brown/grey gradient. Healthy: PSNR ~20+, LPIPS ≲0.4, structure
  visible.
- **`cap_max` 1M is a proven *optimum*, not just a ceiling.** Full sweep on
  the 1318-image `lr_full` scene: 500k = LPIPS 0.394 (valid but worse),
  **1M = best**, 2M = fog (LPIPS 0.91), 3M = fog (LPIPS 0.91). Lower *and*
  higher are both worse — 1M is the sweet spot, not an untested guess.
- **More `--steps` helps (100k > 50k), but NEVER set `--max-steps`.** Proven on
  `lr_full`: 50k = LPIPS 0.384, **100k = 0.365 (~5% better, new best)** — at
  the *same* cap/res. This works *because* the LR schedule denominator
  (`max_steps`) stays clamped at 30000, so steps past 30k are gentle frozen-LR
  refinement. Setting `--max-steps 100000` (stretching the schedule to match
  `--steps`) keeps LR + noise-injection high too long → **fog (LPIPS 0.92)**.
  The hardcoded 30000 default is load-bearing; treat `--max-steps` as
  do-not-touch.
- **`images_2` (higher res) fogs — also proven.** `lr_full` at `images_2`/1M
  →  fog (PSNR 12.9, LPIPS 0.92). Higher res needs more splats to fit, but more
  splats *also* fog dense scenes — mutually exclusive, no viable `cap_max`
  exists. `images_4` is the only stable training tier here.
- **There is a real per-scene detail ceiling.** On a large scene, soft
  large surfaces (e.g. a couch) at the stable config are *not* a tuning miss —
  pushing resolution or budget to fix them destroys the whole reconstruction.
  A smaller scene reconstructs sharper than a big one at the same fixed budget.
- **16 GB host RAM is the main constraint** — full-resolution COLMAP and
  large all-image full-res training can OOM/crash the machine.

See [`memory.md`](memory.md) for the full benchmarking table, every dataset's
sharpness, and the complete experiment history.

---

## Building VkSplat (the underlying trainer)

The trainer is a Vulkan compute extension with a Python binding.

### Prerequisites
- Vulkan SDK installed
- Python 3.7+
- C++17 compiler; pybind11/setuptools **and/or** CMake 3.16+

Tested with Vulkan 1.3/1.4 on Windows 10/11 and Ubuntu 22.04+, across NVIDIA,
AMD (incl. RX 7900 XTX), and Intel GPUs.

### Install (pip — recommended)
```bash
cd /path/to/vksplat
pip install -e . --no-build-isolation   # add -v for verbose
```
```python
import vksplat
```

### Install (CMake — for development)
```bash
# Linux
cd /path/to/vksplat && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
# Windows
cd /path/to/vksplat
cmake -B build && cmake --build build --config Release
```
Ensure `numpy`, `opencv-python`, `tqdm` are installed for the trainers;
`torchmetrics[image]>=1.0.1` for evaluation (no CUDA GPU required for eval).

### Recompile shaders / C++
```bash
python3 compile_shaders.py            # add --force to ignore cache
pip install -e . --no-build-isolation -v   # rebuild C++ (delete build/ if cache stale)
```
If you see "Shaders must be compiled with USE_XXX=1", adjust
`USE_EMULATED_INT64` / `USE_EMULATED_F32_ATOMIC` in
`vksplat/slang/config.slang` and recompile shaders.

Source layout: Slang in `vksplat/slang/`, compiled SPIR-V in
`vksplat/shader/generated/`, Vulkan/C++ in `vksplat/src/`
(`buffer`, `gs_pipeline`, `gs_renderer`, `gs_trainer`).

---

## Citation

```bibtex
@inproceedings{chen2026vksplat,
  booktitle = {Eurographics 2026 - Short Papers},
  title     = {{VkSplat: High-Performance 3DGS Training in Vulkan Compute}},
  author    = {Chen, Jingxiang and Ibrahim, Mohamed and Liu, Yang},
  year      = {2026},
  publisher = {The Eurographics Association},
  ISSN      = {2309-5059},
  ISBN      = {978-3-03868-299-8},
  DOI       = {10.2312/egs.20261024}
}
```
