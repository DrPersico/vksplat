# VkSplat 3DGS Training — Project Memory

## System Specs
- **GPU (CURRENT, 2026-05-18)**: **AMD Radeon RX 7900 XTX (24 GB) — REINSTALLED.** The GTX 1070 Ti is **REMOVED**. ROCm torch (`2.9.1+rocmsdk`, default `python`) now sees it: `cuda.is_available()==True`, device 0 = "AMD Radeon RX 7900 XTX". **NAFNet `restore_frames.py --method dl` runs on it via the DEFAULT `python`** (~0.2 img/s full-res, ~2.5× the 1070 Ti). VkSplat training is possible again.
  - **The `E:\nvidia-gsplat\python312` CUDA env is now DEFUNCT** — no NVIDIA card present. Do NOT use it. Run everything (deblur, train) with the default `python`. (History below kept for context only; the "use the NVIDIA env" instructions in Issue #11 are SUPERSEDED.)
  - B450 Gaming Plus has only two slots; 7900 XTX + 1070 Ti never coexisted. Card has been swapped back and forth this project — always re-probe `torch.cuda.is_available()` at session start; do not trust prior session's GPU assumption.
  - *(Historical: 2026-05-17 the XTX was out and a 1070 Ti was in for a gsplat attempt — gsplat dead-ended on Pascal sm_61. See DEAD END section + Issue #11.)*
- **RAM**: 16 GB host memory (major constraint!)
- **OS**: Windows 10
- **Phone**: Samsung (4000×1848 raw pixels, EXIF orientation 6)

### NVIDIA gsplat environment (set up 2026-05-17)
- **Isolated Python**: `E:\nvidia-gsplat\python312\python.exe` — Python 3.12.8 **embeddable** package (off C: per constraint). Fully independent of the C: ROCm-torch Python (`C:\Users\Persico\AppData\Local\Programs\Python\Python312`, which has `torch 2.9.1+rocm7.2.1` — DO NOT disturb; reserved for 7900 XTX/VkSplat return).
- Embeddable has **no `venv`/`ensurepip` module** (stripped by design). pip bootstrapped via get-pip.py (pip 26.1.1). `python312._pth` edited: added `Lib\site-packages` + uncommented `import site` so pip-installed packages import. Used **directly as the env** (no venv — it's already a dedicated isolated interpreter).
- Existing `E:\brush_venv` is a ROCm-torch venv (torch 2.9.1+rocm7.2.1) from earlier Brush/gsplat-ROCm work — NOT for NVIDIA, ignore.
- **CUDA torch**: `torch 2.5.1+cu121` + `torchvision 0.20.1+cu121` installed & VERIFIED (cuda.is_available True, GTX 1070 Ti, capability (6,1)=sm_61 in arch_list, real GPU matmul OK).
- **CUDA toolkit**: nvcc 12.6 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`. **MSVC**: VS2022 BuildTools `cl.exe` 14.44.35207 (also VS2019 14.27 fallback). vcvars64.bat at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`.

### Building gsplat on Windows + GTX 1070 Ti — the working recipe (2026-05-17)
Build via `E:\nvidia-gsplat\build_gsplat.bat` (calls vcvars64, sets `TORCH_CUDA_ARCH_LIST=6.1` + `DISTUTILS_USE_SDK=1` + `MAX_JOBS=6`, pip installs `-e E:\nvidia-gsplat\gsplat`, logs full output to `E:\nvidia-gsplat\build_full.log`). **Five distinct blockers, each hiding the next — fix ALL of them:**
1. `error: command 'cl.exe' failed` → MSVC not on PATH. Fix: build inside `vcvars64.bat` env (the .bat does `call`).
2. `ValueError: Unknown CUDA arch ()` → `TORCH_CUDA_ARCH_LIST` lost through inline `cmd /c ... && ...` chaining. Fix: set it inside a standalone .bat (not inline).
3. `fatal error C1083: Cannot open 'glm/gtc/type_ptr.hpp'` → shallow `git clone` skipped the bundled GLM **submodule**. Fix: `git submodule update --init --recursive` in `E:\nvidia-gsplat\gsplat`.
4. `ivalue_inl.h: error: type name is not allowed` (nvcc parsing torch headers) → **gsplat 1.5.x forces `-std=c++20`; nvcc 12.6's cudafe++ cannot parse torch 2.5.1 headers under C++20.** Fix: **use gsplat 1.4.0** (`git checkout v1.4.0` + re-run submodule update). 1.4.0's setup.py sets NO explicit C++ std → defaults to C++17, which works. 1.4.0 still has full MCMC + COLMAP trainer.
5. `python_headers.h: fatal error C1083: Cannot open 'Python.h'` → embeddable Python has NO dev headers/libs. Fix: copy `include\` and `libs\` from the C: Python 3.12.8 (EXACT same version, ABI-safe; read-only copy OUT of C:, doesn't disturb ROCm Python) into `E:\nvidia-gsplat\python312\`.
gsplat has no prebuilt Windows wheels (`docs.gsplat.studio/whl/*` is Linux-only) — must compile from source.

### ❌ VERDICT: gsplat does NOT support the GTX 1070 Ti (Pascal sm_61) — DEAD END (2026-05-17)
After fixing all 5 environment blockers above, build #6 reached gsplat's own kernels and hit a **6th, UNFIXABLE blocker — a hardware floor, not an environment issue:**
- Error: `namespace "cooperative_groups" has no member "labeled_partition"`
- `cg::labeled_partition` is a CUDA cooperative-groups primitive requiring **Compute Capability ≥ 7.0 (Volta)**. The GTX 1070 Ti is **sm_61 (Pascal, 2017)** — the silicon physically lacks it.
- Used in **8 sites across 6 BACKWARD-pass kernels** (`fully_fused_projection_bwd.cu`, `..._2dgs_bwd.cu`, `..._packed_bwd.cu`, `..._packed_2dgs_bwd.cu`, `world_to_cam_bwd.cu`). These are the gradient kernels training depends on — not optional.
- **The earlier "gsplat works on Pascal sm_61" claim was WRONG.** torch/arch-list supporting sm_61 is necessary but NOT sufficient — gsplat's CUDA kernels assume Volta+. gsplat requires sm_70+ regardless of OS/toolchain.
- Options considered & rejected: (A) patch all 8 `labeled_partition` sites with a `coalesced_threads` emulation — high effort, silent-wrong-gradient risk; (B) even older gsplat — loses the quality features that motivated the switch, may hit other Pascal blockers. **User chose: STOP. gsplat needs Volta+; the 1070 Ti is below the floor.**
- **Net hardware reality on this machine: NEITHER good 3DGS path exists right now.** VkSplat needs the (removed) AMD 7900 XTX; gsplat needs an NVIDIA Volta+ (the installed 1070 Ti is too old). To use gsplat would need a ≥sm_70 NVIDIA card (RTX 20-series / GTX 1650+ Turing or newer). To use VkSplat, reinstall the 7900 XTX.
- **Salvaged asset**: the GPU-COLMAP work is NOT wasted — `run_colmap.py --gpu` + CUDA colmap.exe is GPU-agnostic, reusable for any future 3DGS run (VkSplat or gsplat). The lr_full reconstruction COLMAP run completed on the 1070 Ti and is staged for whenever a working trainer is available.
- The `E:\nvidia-gsplat\` env (Python 3.12.8 + torch cu121 + gsplat 1.4.0 source + build recipe) is left intact — it would work immediately if a Volta+ NVIDIA card is installed (only the GPU is the blocker).

### 🐍 Python environments on this machine — CONSOLIDATED REFERENCE (which torch, what for)

> **⚠️ 2026-05-18 UPDATE — read first:** Only the **RX 7900 XTX** is installed now (1070 Ti out). Use the **default `python`** (ROCm) for EVERYTHING — it now sees the 7900 XTX (`cuda.is_available()==True`). The `E:\nvidia-gsplat\python312` CUDA env is **DEFUNCT** (no NVIDIA card). The table below is the older multi-card history; the "currently NONE / probe fails" notes for the default python are NO LONGER TRUE (that was while the XTX was physically out).

This machine has MULTIPLE Pythons with DIFFERENT torch builds. Using the wrong one silently picks the wrong GPU backend (e.g. ROCm torch can't see an NVIDIA card → CPU fallback). Always pick by the table:

| Interpreter | torch | GPU it sees | Use for |
|---|---|---|---|
| `C:\Users\Persico\AppData\Local\Programs\Python\Python312\python.exe` (the **default `python`** on PATH) | `2.9.1+rocm7.2.1` (AMD ROCm) | RX 7900 XTX (when installed) — currently NONE (probe fails: "Failed to get device count") | Day-to-day scripts, VkSplat/AMD path. **DO NOT disturb its torch** — reserved for the 7900 XTX return. |
| `E:\nvidia-gsplat\python312\python.exe` (Python 3.12.8 **embeddable**, isolated) | `2.5.1+cu121` (NVIDIA CUDA) | **GTX 1070 Ti** (`cuda.is_available()` True, sm_61) | **NAFNet GPU deblur** (`restore_frames.py --method dl`); gsplat (but gsplat is a 1070 Ti dead-end, see verdict above). Has cv2/scipy/numpy/PIL installed. No `venv`/`ensurepip` (embeddable); used directly as the env. |
| `E:\brush_venv\` | `2.9.1+rocm7.2.1` (AMD ROCm) | AMD only | Old Brush/gsplat-ROCm experiment. **Ignore** — not for NVIDIA, superseded. |

- **Rule of thumb**: NVIDIA/CUDA work → `E:\nvidia-gsplat\python312\python.exe`. Everything else (default) → ROCm torch. They are fully independent; installing into one never touches the other.
- **Concrete gotcha proven 2026-05-17**: `python restore_frames.py --method dl` (default Python) → ROCm torch → probe fails → CPU fallback (~12 h for 900 frames). `E:\nvidia-gsplat\python312\python.exe restore_frames.py --method dl` → CUDA torch → 1070 Ti → minutes. Same script, 100× difference, purely from interpreter choice.
- The 1070 Ti is fine for **inference** (NAFNet) but NOT gsplat **training** (gsplat needs Volta+ backward kernels; see DEAD END verdict). VkSplat training still needs the (removed) 7900 XTX. See Issue #11 for the deblur-salvage details.

---

## What Works

### VkSplat Installation
- Built from source with pybind11 + custom Vulkan SDK setup
- Vulkan SDK: manually cloned `KhronosGroup/Vulkan-Headers`, created `E:\VulkanSDK` structure
- Requires Visual Studio Build Tools (installed via `winget`)
- `pip install -e .` in `E:\vksplat\vksplat\`

### COLMAP Pipeline (`run_colmap.py`, `run_colmap_sequential.py`)
- Uses `pycolmap` Python API (not CLI)
- **Must run on `images_2` (half-res)** — full-res COLMAP crashes PC (RAM)
- Settings that work:
  - `camera_model = "SIMPLE_RADIAL"`
  - `CameraMode.SINGLE` for same-camera photos
  - `CameraMode.PER_FOLDER` for mixed cameras (photos + video)
  - `sift.first_octave = 0` (prevents 4× RAM upscale)
  - `sift.max_num_features = 8192`
  - `num_threads = 4`
  - `device = pycolmap.Device.cpu`
  - **Exhaustive matching** (`run_colmap.py`) — best for unordered photo sets
  - **Sequential matching** (`run_colmap_sequential.py`) — for sequentially-captured / timestamped photos
    - `SequentialPairingOptions(overlap=10, quadratic_overlap=True)`
    - O(N) instead of O(N²) — on 900 sequential photos: ~22 min matching vs hours for exhaustive
    - Use when filenames are timestamps or frames are visually adjacent in capture order
- **pycolmap on this install has NO GPU SIFT support** — error: `"Cannot use GPU feature extraction without CUDA or OpenGL support"`. `Device.cuda` fails the option check before extraction starts. All COLMAP stages run on CPU regardless. Reason: pycolmap binary was not compiled with `OPENGL_ENABLED` / `CUDA_ENABLED`. (The 7900 XTX is AMD anyway, so CUDA wouldn't help.)
- **GPU COLMAP path (added 2026-05-17, for when GTX 1070 Ti is installed):**
  - Standalone CUDA COLMAP 4.0.4 downloaded + extracted to `E:\vksplat_tools\bin\colmap.exe` (62 MB exe; full Qt/CUDA dist in `E:\vksplat_tools\bin\`; zip kept at `E:\vksplat_tools\colmap-cuda-4.0.4.zip`). **Off C: per project constraint.**
  - `run_colmap.py` gained `--gpu` / `--colmap-exe <path>` / `--gpu-index <n>` flags. `--gpu` shells out to CUDA `colmap.exe` for **stage 1 (feature_extractor) + stage 2 (exhaustive_matcher)** only. Stage 3 (incremental mapping) ALWAYS stays on pycolmap CPU — COLMAP's mapper has no GPU path upstream.
  - GPU mode mirrors the CPU config: `--ImageReader.single_camera 1`, `SIMPLE_RADIAL`, `first_octave 0`, `max_num_features 8192`, optional `--camera-params` → `--ImageReader.camera_params`.
  - Default no-`--gpu` path is byte-for-byte the old pycolmap CPU behavior (unchanged; safe fallback).
  - Expected speedup: O(N²) matching on 1321 imgs drops from ~8 h CPU → ~20–40 min on the 1070 Ti.
  - Launch once 1070 Ti is CUDA-visible: `python E:\vksplat\run_colmap.py E:\vksplat_data\lr_full --image-subdir images_2 --camera-params 1343.6,1000,462,0.051 --gpu`
  - B450 Gaming Plus: 7900 XTX → PCIE_1 (CPU x16, full), 1070 Ti → PCIE_4 (chipset PCIe 2.0 x4). Top slot NOT split by populating the second; x4 is fine for SfM (not bandwidth-bound). Watch physical clearance under the 7900 XTX + PSU 8-pin count (~600 W combined load → 750 W+ PSU).
- When COLMAP produces wrong focal length (e.g. f=3044, k=1.54), fix by providing initial camera params from a known-good run: `reader_opts.camera_params = '1344,1000,462,0.05'`

### Image Preparation (`prepare_dataset.py`)
- **EXIF stripping is critical**: use `PIL.Image.open()` → `np.array()` to get raw pixels WITHOUT rotation, then resize with OpenCV
- **Apply EXIF orientation manually** to the raw pixels (so saved images are upright AND consistent for COLMAP). Samsung phones in portrait capture come out as `orientation=6` (rotate 270° CW) on a landscape sensor buffer
- Create three tiers: `images/` (full), `images_2/` (half), `images_4/` (quarter)
- All saved as JPEG quality 95 with NO EXIF
- `prepare_dataset.py` takes explicit `--width-2` / `--width-4` flags rather than dividing by 2/4 — the "images_2" naming is a *convention* (HD-ish), not a literal fraction of source. For 8K sources the right tier widths are ~2040 / 1020, not 4080 / 2040
- Sequential per-image processing handles arbitrary EXIF orientations via `apply_orientation()` helper supporting all 8 EXIF cases

### Training Configuration (Best Results)
- **MCMC strategy** is far superior to default (controls splat count)
- Best config for 2000×924 images:
  - `strategy = "mcmc"`
  - `cap_max = 1_000_000` (sweet spot for quality vs VRAM)
  - `train_steps = 30_000`
  - `ssim_lambda = 0.2`
  - VRAM usage: ~1.0 GB
  - Training time: 4-5 minutes

### Evaluation (`run_eval.py`)
- **Must run in a subprocess** to free training VRAM/RAM first
- Processes images one-by-one with `torch.cuda.empty_cache()` + `gc.collect()`
- Uses `torchmetrics` on GPU (PSNR, SSIM, LPIPS)

### Full-Resolution Training
- COLMAP run on `images_2` → intrinsics for 2000×924
- To train on full-res `images` (4000×1848), must create `sparse/0_fullres` with 2× scaled intrinsics:
  - `f *= 2`, `cx *= 2`, `cy *= 2`, `width *= 2`, `height *= 2`
  - Distortion `k` stays the same

### Does higher-res COLMAP improve quality? NO (answered 2026-05-17)
- COLMAP at `images_2` (2000×924) already registers 99%+ with clean intrinsics
  (f≈1351, k≈0.04) — pose accuracy is NOT the quality bottleneck.
- Rendered quality is capped by the TRAINING-resolution / MCMC-fog ceiling
  (images_4@1M is the proven max; images_2 fogs — see 5-lever sweep), NOT by
  COLMAP. Better COLMAP poses feed a training step that's already capped.
- vksplat loader rescales intrinsics from any tier → any training tier anyway.
- Only marginal effect of hi-res COLMAP: slightly denser sparse seed cloud;
  but images_2 seed density already sufficient (training converged to ceiling),
  and full-res COLMAP risks the documented RAM crash. NOT worth it.
- **Quality levers in order: sharp capture (HUGE) > training config Exp C
  (~9%) > COLMAP resolution (negligible here). Invest in DATA, not COLMAP res.**

### Intrinsic Rescaling — How VkSplat Actually Handles It
- The C++ loader at `vksplat/src/gs_trainer.cpp:319-374` has TWO layers:
  1. **Hardcoded MipNeRF360 factor map** (lines 321-341): if `image_dir` ends in `images_2`/`images_4`/`images_8`, divides intrinsics by 2/4/8 — assumes sparse is at full source resolution
  2. **Safety net** (lines 367-374): if loaded image dims don't match `camera.w/h` after factor adjustment, rescales `fx/fy/cx/cy` by actual-vs-camera ratio
- **Net effect**: you can train at *any* image tier against COLMAP intrinsics from *any* other tier — the safety net corrects the discrepancy. E.g., COLMAP on images_2 (2040 wide) + training on images_4 (1020 wide): factor map divides by 4 → safety net multiplies by 2 → correct `f * 1020/2040`
- This means `sparse/0_fullres` with manually-scaled intrinsics is **not strictly required** for cross-resolution training — but it's still useful when COLMAP intrinsics don't match any of the factor-map keys

---

## Quality Results

| Dataset | Images | Resolution | Strategy | Splats | PSNR | SSIM | Time |
|---------|--------|-----------|----------|--------|------|------|------|
| Kitchen photos | 85 | 2000×924 | MCMC 1M | 1M | 21.76 | — | 3.6 min |
| Kitchen photos | 85 | 2000×924 | Default | 14.8M | 17.40 | — | — |
| Living room photos | 246 | 2000×924 | MCMC 1M | 1M | **24.38** | 0.615 | 4.5 min |
| Combined (photos+video) | 720 | mixed | MCMC 1M | 1M | 18.35 | 0.699 | ~8 min |
| Phone photos (pre-EXIF fix) | 84 | 2000×924 | Default | 9.5M | 9.85 | — | — |
| Phone photos (post-EXIF fix) | 84 | 2000×924 | Default | 9.5M | 18.44 | — | — |
| LR subset full-res | 83 | 4000×1848 | MCMC 2M | 2M | 18.39 | 0.467 | 26.4 min |
| Full dataset (smoke) | 891 reg / 900 | 1020×2209 | MCMC 0.5M | 500k | 20.87 | 0.798 | 0.7 min (5k steps) |
| Full dataset (2M/30k) | 891 reg / 900 | 1020×2209 | MCMC 2M | 2M | 13.18 | 0.695 | ~10 min — DEGENERATE (over-densified to fog) |
| Full dataset (1M/30k) | 891 reg / 900 | 1020×2209 | MCMC 1M | 1M | 24.16 | 0.841 | BLURRY source — high PSNR is misleading (blur matches blur) |
| **sharp / new fotos (1M/30k)** | 579/581 | 1000×462 | MCMC 1M | 1M | 21.57 | 0.682 | **SHARP source — lower PSNR but LPIPS 0.36 (better); renders show real texture/parquet/fabric. THE GOOD ONE.** 155s |

### Key Findings
1. **More photos = better**: 249 photos (PSNR 24.38) vs 85 photos (PSNR 21.76) — significant improvement
2. **MCMC >> Default strategy**: Default over-densifies to millions of splats with worse quality
3. **Video frames hurt quality**: Combined dataset (18.35) worse than photos-only (24.38) — compression artifacts, motion blur, and diluted splat budget
4. **cap_max 1M vs 2M**: Similar quality (21.76 vs 21.59 on kitchen) — suggesting a quality ceiling

---

## 📊 BENCHMARKING REFERENCE — consolidated results (all datasets, as of 2026-05-17)

**Engine for all rows below = VkSplat (Vulkan/AMD RX 7900 XTX, 24 GB).** gsplat never produced a result on this machine (Pascal 1070 Ti below sm_70 floor — see DEAD END section). All numbers are VkSplat. PSNR/SSIM are unreliable on blurry sources (blur-matches-blur inflates them) — **for cross-dataset comparison use LPIPS + visual inspection**, not PSNR.

### A. Training runs (every recorded result, best→worst within group)

| # | Dataset | Imgs (reg) | Train res | Strategy / cap_max | Steps | PSNR | SSIM | LPIPS | Verdict |
|---|---------|-----------|-----------|--------------------|-------|------|------|-------|---------|
| 1 | Living room photos (old) | 246 | 2000×924 | MCMC 1M | 30k | **24.38** | 0.615 | — | High PSNR but pre-sharp-era; soft |
| 2 | Full dataset (1M) | 891/900 | 1020×2209 | MCMC 1M | 30k | 24.16 | 0.841 | 0.42 | **BLURRY source — PSNR misleading** |
| 3 | Full dataset (2M) | 891/900 | 1020×2209 | MCMC 2M | 30k | 13.18 | 0.695 | — | DEGENERATE fog (over-densified) |
| 4 | Full dataset (3M) | 891/900 | 1020×2209 | MCMC 3M | 30k | — | — | — | AMD driver TDR hang (unrunnable) |
| 5 | Full dataset (smoke) | 891/900 | 1020×2209 | MCMC 0.5M | 5k | 20.87 | 0.798 | — | Clean smoke baseline |
| 6 | **sharp / "new fotos"** ⭐ | 579/581 | 1000×462 (img_4) | MCMC 1M | 30k | 21.57 | 0.682 | **0.36** | **BEST REAL RESULT — real texture/parquet/fabric. Baseline to beat.** |
| 7 | sharp / "new fotos" (Exp A) | 579/581 | 1000×462 | MCMC 1M, ssim_λ 0.2 | 50k | ~21.5 | — | ~0.34 | Exp A baseline |
| 8 | **sharp (Exp C)** ⭐ | 579/581 | 1000×462 | MCMC 1M, **ssim_λ 0.4** | **50k** | — | — | **0.330** | **BEST CONFIG — ~9% better LPIPS than Exp A. Use this.** |
| 9 | sharp (Exp D) | 579/581 | 1000×462 | MCMC 1M, ssim_λ 0.4 + tuned reg | 50k | — | — | ~0.33 | ≈ Exp C, no clear win |
| 10 | sharp @ images_2 (hi-res) | 579/581 | 2000×924 | MCMC 1M | 30k | 12.45 | — | 0.84 | FOG (`~990k relocate/0 add`) — hi-res needs lower cap_max |
| 11 | sharp img_2 + refine_stop 12k | 579/581 | 2000×924 | MCMC 1M | 30k | 14.08 | — | 0.87 | Still fog — refine_stop doesn't fix |
| 12 | sharp img_2 + opacity_reg .05 (Exp B) | 579/581 | 2000×924 | MCMC 1M | 30k | 9.74 | — | 0.838 | WORSE — opacity_reg destabilized |
| 13 | full2 (low-sharpness test) | 1361/1380 | 1000×462 | MCMC 1M | 30k | 12.0 | — | 0.88 | DEGENERATE — blurry source confirmed fatal |
| 14 | Kitchen photos | 85 | 2000×924 | MCMC 1M | 30k | 21.76 | — | — | Small-scene baseline |
| 27 | **Kitchen A/B — Run A (clean baseline)** | 85 | 2000×924 | MCMC 1M | 30k | **22.52** | **0.644** | **0.444** VGG / **0.441** Alex | A/B baseline (images_4). eval.json `E:\vksplat_output\kitchen_ab_clean\20260518_170813_kitchen\eval.json` |
| 28 | **Kitchen A/B — Run B (restored, vs restored GT)** | 85 | 2000×924 | MCMC 1M | 30k | 22.27 | 0.582 | 0.457 VGG / **0.484** Alex | Restored renders vs restored GT (harder target). eval.json `E:\vksplat_output\kitchen_ab_restored\20260518_171134_kitchen\eval.json` |
| 29 | **Kitchen A/B — Run B renders vs blurry GT** | — | — | — | — | **22.63** | **0.636** | **0.436** VGG / **0.427** Alex | Cross-eval: Run B renders vs Run A's blurry GT. **BEST LPIPS of all three comparisons** (0.427 < 0.441 Run A) — deblurred 3D model is perceptually closer to blurry reality than clean baseline |
| 15 | Kitchen photos | 85 | 2000×924 | Default | 30k | 17.40 | — | — | Default strat → 14.8M splats, worse |
| 16 | Combined (photos+video) | 720 | mixed | MCMC 1M | 30k | 18.35 | 0.699 | — | Video frames hurt quality |
| 17 | LR subset full-res | 83 | 4000×1848 | MCMC 2M | 30k | 18.39 | 0.467 | — | Few imgs + full-res = worse |
| 18 | Phone photos (pre-EXIF fix) | 84 | 2000×924 | Default | 30k | 9.85 | — | — | EXIF orientation bug (issue #1) |
| 19 | Phone photos (post-EXIF fix) | 84 | 2000×924 | Default | 30k | 18.44 | — | — | After EXIF fix |
| 20 | **lr_full** (1321 photos) | **1318/1321 (99.8%)** | 1000×462 (img_4) | MCMC 1M, ssim_λ 0.4 | 50k | **20.82** | **0.673** | **0.384** (Alex; VGG 0.397) | **VALID but NOT a new best 2026-05-18 (Exp C config). 255,523 COLMAP pts, 165 val. Render visibly SHARP (figurine/wood-grain/framed-art crisp), ZERO fog — but LPIPS-Alex 0.384 is ~16% WORSE than the 0.330 sharp/new-fotos baseline. eval.json + splat.ply (236MB) at `E:\vksplat_output\lr_full_lr_full_best\20260518_085211_lr_full\`.** |
| — | lr_full analysis | — | — | — | — | — | — | — | **Larger scene (1318 img / 255k pts / wide extent) vs 579-img sharp set → same 1M splat budget spread thinner → lower per-region detail → worse LPIPS. ⭐ BEST living-room result remains sharp/"new fotos" Exp C, LPIPS 0.330 (rows 6/8).** |
| 21 | **lr_full img_4 @ cap_max 2M** (Run A) | 1318/1321 | 1000×462 | MCMC **2M**, ssim_λ 0.4 | 50k | **12.33** | 0.453 | **0.906** (Alex; VGG 0.662) | ❌ **FOG — DISPROVES the "higher cap_max helps lr_full" hypothesis. 1M→2M destroyed it (val_74 = uniform brown gradient, was crisp parquet at 1M). Confirms issue #8: lr_full's healthy budget is ≤1M, NOT above. Denser scene does NOT want more splats. Output `E:\vksplat_output\lr_full_lr_full_img4_2M\20260518_092520_lr_full\`.** |
| 22 | **lr_full img_2 @ cap_max 1M** (Run B) | 1318/1321 | 2000×924 | MCMC 1M, ssim_λ 0.4 | 50k | **12.88** | 0.470 | **0.921** (Alex; VGG 0.691) | ❌ **FOG — DISPROVES the resolution hypothesis. 2× pixels at 1M = total fog (val_74 smeared, was crisp parquet at img_4/1M). img_2 needs more splats to fit detail, but more splats ALSO fog this scene (Run A) → mutually exclusive, no viable cap_max exists for img_2 on lr_full. Output `E:\vksplat_output\lr_full_lr_full_img2_1M\20260518_133614_lr_full\`.** |
| 23 | **lr_full img_4 @ cap_max 500k** (Arm C) | 1318/1321 | 1000×462 | MCMC **500k**, ssim_λ 0.4 | 50k | 20.79 | 0.669 | **0.394** (Alex; VGG 0.404) | ✅ Valid (no fog), couch render ≈ baseline by eye. **But marginally WORSE than 1M (0.394 vs 0.384) on every metric. DISPROVES the issue-#8 "dense scene prefers FEWER splats" hypothesis for lr_full — 1M is a genuine optimum, not an untested-downward assumption. cap_max curve so far: 500k=0.394, 1M=0.384(best), 2M=fog, 3M=pending. Output `E:\vksplat_output\lr_full_lr_full_500k\20260518_151258_lr_full\`.** |
| 24 | **lr_full plain-100k** (Arm A) ⭐ | 1318/1321 | 1000×462 | MCMC 1M, ssim_λ 0.4, max_steps 30k | **100k** | **21.16** | **0.679** | **0.365** (Alex; VGG 0.385) | ✅ **NEW lr_full BEST. 100k steps (LR still clamps at 30k → steps 30k-100k at frozen final LR) beat 50k baseline on ALL metrics (LPIPS 0.365 vs 0.384, ~5%). Couch render visibly sharpest yet (velvet pile/sheen/ribbing). DISPROVES the code-based "frozen-LR steps are inert" prediction — slow refinement + noise over 70k extra steps measurably helps. Answers user Q1: YES, 100k improves lr_full. Output `E:\vksplat_output\lr_full_lr_full_100k_plain\20260518_152948_lr_full\`.** |
| 25 | **lr_full stretched-100k** (Arm B) | 1318/1321 | 1000×462 | MCMC 1M, ssim_λ 0.4, **--max-steps 100000** | 100k | **12.58** | 0.455 | **0.924** (Alex; VGG 0.656) | ❌ **FOG. Identical to Arm A EXCEPT max_steps 30k→100k → total degeneration (val_74 = brown smear). Stretching the LR schedule keeps LR + noise-injection magnitude (`noise_lr*get_means_lr`, gs_trainer.cpp:1274) high for ~60k+ steps → prolonged relocation churn → fog (same mechanism class as cap_max fog, triggered by too-slow schedule). Also = the definitive `--max-steps` wiring test: A(30k)=0.365 vs B(100k)=0.924, drastically different → flag PROVEN functional. Output `E:\vksplat_output\lr_full_lr_full_100k_stretch\20260518_154952_lr_full\`.** |
| — | 🔑 KEY FINDING — steps vs schedule (2026-05-18) | — | — | — | — | — | — | — | **More steps help ONLY with `max_steps` clamped. plain-100k (max_steps stays 30k) = BEST (LPIPS 0.365). stretched-100k (max_steps=100k) = FOG (0.924). The hardcoded `max_steps=30000` in simple_trainer.py:53 is LOAD-BEARING, not a quirk — it's what makes long runs safe by freezing LR/noise after step 30k so extra steps are gentle refinement, not churn. RECOMMENDATION: for more quality, raise `--steps` (e.g. 100k) but NEVER pass `--max-steps` > ~30k. Earlier "50k optimal" belief corrected by experiment.** |
| 26 | **lr_full img_4 @ cap_max 3M** (Arm D) | 1318/1321 | 1000×462 | MCMC **3M**, ssim_λ 0.4 | 50k | **12.20** | 0.452 | **0.913** (Alex; VGG 0.668) | ❌ **FOG (val_74 = featureless tan). 3M smoke (2k steps) survived clean — did NOT TDR-hang (prior 3M hang was full-res, different dataset) — but full 50k fogs like 2M. Completes the cap_max curve.** Output `E:\vksplat_output\lr_full_lr_full_3M\20260518_161115_lr_full\`. |
| — | 🔚 lr_full SWEEP COMPLETE — DEFINITIVE (2026-05-18) | — | — | — | — | — | — | — | **lr_full BEST CONFIG = 1M / images_4 / ssim_λ 0.4 / `--steps 100000` (NO `--max-steps`) → PSNR 21.16, LPIPS-Alex 0.365 (Arm A, row 24). Full proven curves: **cap_max** {500k=0.394 valid, **1M=best**, 2M=fog 0.906, 3M=fog 0.913} — 1M is a true optimum, both directions worse. **steps/schedule** {50k=0.384, **100k-plain=0.365 BEST**, 100k-stretched(max_steps=100k)=fog 0.924}. KEY: more steps help ONLY with max_steps clamped at 30k (load-bearing default, simple_trainer.py:53). ↑resolution (img_2)=fog. Soft couch is the scene detail ceiling at the stable optimum — improved ~5% by 100k-plain but not eliminated. ⭐ Overall champion still sharp/"new fotos" Exp C LPIPS 0.330 @50k (rows 6/8); HIGHEST-VALUE FUTURE TEST = re-run champion @ `--steps 100000` (no --max-steps) — the 100k-plain gain likely transfers and could push below 0.330.** |
| — | ⚠️ densification-log lesson | — | — | — | — | — | — | — | **`0 add → 1M splats` from ~step 10k = HEALTHY early budget-saturation on a rich scene, NOT issue #8 fog (fog = PSNR 9–14 / LPIPS 0.84+).** Agent raised 2 FALSE fog alarms here from the log before eval, even prompting a user decision on a non-existent failure. **Rule reaffirmed: judge ONLY by final eval LPIPS + rendered image; `0 add` alone is ambiguous and must NOT be called fog from the log. Do not repeat.** |

⭐ = the result to cite for "best living-room reconstruction". **Definitive baseline = row 6/8 (sharp dataset, MCMC 1M @ images_4, ssim_λ 0.4, 50k, LPIPS 0.330).** Paths: data `E:\vksplat_data\sharp\`, output `E:\vksplat_output\sharp_mcmc1m\`.

### B. Source-dataset sharpness inventory (var-of-Laplacian; >300 sharp, 150–300 usable, <150 too blurry)

| Dataset (path) | Photos | Median sharp | % usable | Segments | Use? |
|----------------|--------|--------------|----------|----------|------|
| **sharp / "new fotos"** `E:\vksplat_data\sharp\` | 581 | **386** | 91% | 12/12 | ✅ **LR winner** |
| **bedroom normal camera** `E:\Downloads\bedroom normal camera` | 474 | **345** | 90% | 12/12 | ✅ equal-quality bedroom |
| **lr_full** `E:\Downloads\living room normal camera fulldataset` → `E:\vksplat_data\lr_full` | 1321 | **344** | 88% | 12/12 | ✅ best LR data, COLMAP in progress |
| full_dataset (900 photos) | 900 | ~15 | 0% | — | ❌ uniformly motion-blurred |
| full2 (1380 photos) | 1380 | 98 | low | — | ❌ too blurry (registered fine, trained to fog) |
| 4K60 / 4K30 videos (4 files) | frames | 15, max 30 | 0% | — | ❌ long-exposure blur, frame-rate irrelevant |
| Rejected variants (multifocus / locked-shutter / RAW / 60fps, same room) | — | all <150 | — | — | ❌ gimmicks lost to plain stop-and-shoot |

**One-line takeaway for benchmarking:** plain default phone camera + deliberate **stop-and-shoot in a bright room** beat every capture gimmick; on sharp data VkSplat MCMC **1M @ images_4 / ssim_λ 0.4 / 50k** is the proven best config (LPIPS 0.330). Hi-res (images_2) fog is intractable on this hardware. Compare future runs by **LPIPS + eye**, never PSNR alone.

---

## Issues Solved

### 1. EXIF Orientation Mismatch (Critical)
- **Symptom**: PSNR ~9-10, complete garbage output
- **Cause**: OpenCV/PIL auto-apply EXIF rotation but COLMAP stores raw pixel coordinates
- **Fix**: Load raw pixels with `PIL.Image.open()` → `np.array()` (no EXIF rotation), then resize with OpenCV

### 2. Host RAM OOM During Evaluation
- **Symptom**: PC freezes, RAM hits 100%
- **Cause**: `torchmetrics` evaluation loading all images + training model in same process
- **Fix**: Run eval in separate subprocess (`run_eval.py`), process images one at a time

### 3. COLMAP RAM Crash on Full-Res Images
- **Symptom**: PC crash during SIFT feature extraction on 4000×1848 images
- **Cause**: `first_octave = -1` (default) internally upscales images 4×, exploding RAM
- **Fix**: Set `first_octave = 0`, run on `images_2`, limit `num_threads = 4`

### 4. COLMAP Fails to Register Images (3/249)
- **Symptom**: Only 3 out of 249 images registered, focal length f=3044, k=1.54
- **Cause**: Bad initial focal length estimate with no prior
- **Fix**: Provide initial camera params from known-good reconstruction: `reader_opts.camera_params = '1344,1000,462,0.05'`

### 5. Combined Dataset Scene Scale Explosion
- **Symptom**: Scene scale = 1,415,050 (should be ~2), PSNR = 4-13
- **Cause**: 2 outlier photos with camera positions at 3-4 million units from centroid
- **Fix**: Filter outlier cameras using `filter_recon.py` (deregister frames with distance > 3× median)

### 6. VkSplat Backward Pass Hardware Limitation
- **Symptom**: `WARNING: a backward implementation is disabled due to hardware limitation`
- **Cause**: `Tensor_0_8_8` backward needs 43 KB shared memory, RX 7900 XTX has 32 KB
- **Impact**: Falls back to `PerSplat` — performance hit only, correctness unaffected

### 7. Over-Densification with Default Strategy
- **Symptom**: 9.5M-14.8M splats, high VRAM, poor quality
- **Cause**: Default strategy densifies without limit
- **Fix**: Switch to MCMC strategy with `cap_max` parameter

### 8. MCMC Over-Densification at HIGH cap_max (full_dataset, May 2026)
- **Symptom**: `full_dataset` 500k/5k smoke = PSNR 20.87 (clean render), but 2M/30k = PSNR 13.18 with completely degenerate output (uniform brown/beige blur, zero scene structure). Same COLMAP, same data, only cap_max + steps changed.
- **Cause**: cap_max too high for the scene's geometric content. MCMC log shows `~2.96M relocate, 0 add` every refine step — it's relocating ~99% of splats each cycle, churning them into a low-opacity fog instead of converging. The 3M run (driver-timeout) showed the same `2962880 relocate, 0 add` pattern.
- **Fix**: LOWER cap_max, not higher. For this dataset 500k was already healthy. Use `cap_max=1_000_000` (memory.md sweet spot) or lower. **Raising cap_max past the scene's healthy budget makes quality catastrophically worse, not just diminishing returns.**
- **Diagnostic tip**: if MCMC logs show `relocate ≈ cap_max, add = 0` consistently, the budget is too high. Healthy MCMC should be adding splats early then stabilizing.
- VRAM also scales super-linearly: 500k=0.67 GB, 2M=12.69 GB.
- **The healthy cap_max is RESOLUTION-dependent (sharp dataset, May 2026)**: `cap_max=1M` was perfect at `images_4` (1000×462, PSNR 21.6, clean) but at `images_2` (2000×924, 4× the pixels) the SAME 1M degenerated to fog (PSNR 12.5, VRAM 12.7 GB, `~990k relocate/0 add` signature — identical to issue #8). Higher training resolution needs a LOWER cap_max, not the same. The fog threshold is a function of (scene geometry × resolution), not an absolute splat number.
- **CONCLUSION — higher resolution is a LOSE on this scene/hardware**: tried images_2 @ 500k (healthy, `~10k relocate/0 add`, VRAM 0.70 GB) to avoid the fog. Result: PSNR 20.5, LPIPS-alex 0.58 — WORSE than images_4 @ 1M (PSNR 21.6, LPIPS 0.36) on every metric AND visually (softer fabric texture, slightly "plasticky"). Why: higher res needs MORE splats to resolve finer detail, but more splats triggers fog, forcing a budget too low to exploit the resolution. Catch-22. **`images_4 @ MCMC cap_max=1M, 30k steps` is this scene's sweet spot on the RX 7900 XTX — do not bother with images_2.** Best output: `E:\vksplat_output\sharp_mcmc1m\20260515_211727_sharp\splat.ply`.
- **refine_stop_iter does NOT fix hi-res fog (tested 2026-05-17)**: images_2 @ 1M with `--refine-stop-iter 12000` (vs MCMC default 25000) → PSNR 14.08, LPIPS-a 0.870, STILL `~960k relocate/0 add`. Stopping the densification refine loop does NOT stop MCMC's per-step relocation/noise churn — separate mechanisms.
- **opacity_reg does NOT fix hi-res fog either (Exp B, 2026-05-17)**: images_2 @ 1M `--opacity-reg 0.05` (vs MCMC default 0.01) → PSNR **9.74** (WORSE than the 12.45 default-fog), LPIPS-a 0.838, still `~835k relocate/0 add`. Raising opacity_reg destabilized optimization further at high res.
- **★ DEFINITIVE: images_2 hi-res fog is INTRACTABLE on this scene/RX 7900 XTX.** Falsified via FOUR independent levers: cap_max sweep (2M/3M fog), refine_stop_iter (fog), cap_max 500k (healthy but worse than images_4), opacity_reg 0.05 (worst). Do NOT retry images_2.
- **★ BEST VkSplat CONFIG (Exp C, 2026-05-17): `images_4 @ MCMC cap_max=1M, ssim_lambda=0.4, 50k steps`.** Progression on sharp `new fotos`: baseline 30k/ssim0.2 LPIPSa 0.362 → Exp A 30k/ssim0.4 0.348 → **Exp C 50k/ssim0.4 0.330** (PSNR 21.66, SSIM 0.691, LPIPSv 0.360). 50k beat 30k on EVERY metric (~9% LPIPS gain vs original, only ~4min extra train, no fog — images_4 is the stable tier). USE THIS CONFIG for all future reconstructions. Command: `train_livingroom.py --image-dir images_4 --strategy mcmc --cap-max 1000000 --steps 50000 --ssim-lambda 0.4`.
- **Exp D closes the hi-res sweep (noise_lr 1e5 @ images_2, 2026-05-17)**: PSNR 20.44, LPIPSa 0.576, NO fog (`~51k relocate` — low noise_lr DID tame churn). But ≈ identical to cap500k (20.45/0.584) — fog-free images_2 is still clearly worse than images_4@1M (21.66/0.330). **Complete images_2 lever sweep (5 levers, zero gaps): cap2M/3M=fog, refstop12k=14.08/fog, opacity_reg=9.74, cap500k=20.45/soft, noise_lr=20.44/soft.** Catch-22 proven from every angle: throttling MCMC enough to avoid fog at hi-res loses more detail than the resolution gains. images_4@1M is the definitive ceiling. VkSplat tuning is EXHAUSTED — remaining quality gains come only from better DATA, not hyperparameters.

### 8c-CORRECTED. 2026-05-19 scene-scope RE-TESTED properly — weak real effect, NOT a 0.20 path
Redid the test with GEOMETRY-box selection (the fix the retraction demanded): central
x-slab of the point cloud → box = **16% of full scene volume** (131 vs 815), kept only
the 21,037 points inside it and the 91 cameras DOMINATED by the box (≥55% of each
camera's observed 3D points inside). Trained Exp C (1M/50k/λ0.4), 12 val.
- **Result: LPIPS-Alex 0.301** (VGG 0.362, PSNR 18.83, SSIM 0.575) vs full-room 0.330.
- **Verdict: scene-scope concentration is a REAL but WEAK lever — ~9% LPIPS gain for a 6×
  scene-volume reduction.** Confirms the budget/volume mechanism in DIRECTION but the
  magnitude is small. Extrapolating, LPIPS 0.20 via scope alone would need impractically
  tiny scenes — NOT a viable path on its own.
- **PSNR/SSIM got WORSE (21.7→18.8, 0.69→0.57):** the cropped 12-view val set is harder
  per-pixel (sparser per-view coverage). So the LPIPS gain is partly val-set-dependent;
  this is "different (smaller) scene, same metric/config", inherently not a pure A/B.
- **Bottom line for capture strategy:** a tighter scene helps modestly; it does not by
  itself reach 0.20. Combined with the 8b exhaustion result, 0.330 stands as the practical
  ceiling for full-room captures on this hardware; ~0.30 is reachable for tightly-scoped
  sub-scenes. Output: `E:\vksplat_output\sharp_box_box91_expC\`. Data: `E:\vksplat_data\sharp_box`.

### 8c-RETRACTED. ⚠️ 2026-05-19 "scene scope" claim WITHDRAWN — flawed subset selection
**The 8c breakthrough below is RETRACTED. Selection was by CAMERA proximity, not scene
geometry. Verified post-hoc: clustering the 30% closest cameras shrank the camera bbox
~3× BUT the reconstructed point-cloud (actual room geometry) only shrank to 80% volume
(815→656) — the clustered cameras still looked across the WHOLE living room. So 1M splats
were NOT concentrated on 1/10 the scene; the scene was ~same size. The 0.330→0.260 gain
is therefore explained by (a) denser redundant view coverage of each surface and (b) an
easier 22-view val set drawn from near the training cluster — NOT by scene-scope/budget
concentration. The "shrink the scene" hypothesis remains UNTESTED. To test it properly:
crop by POINT-CLOUD bbox (keep only points + cameras inside a small spatial box, and only
images whose frustum is dominated by that box), not by camera position. Do not cite 0.260
as a scene-scope result.**

### 8c. ★★ 2026-05-19 BREAKTHROUGH — SCENE SCOPE is the real LPIPS lever  [SEE RETRACTION ABOVE]
After 8b proved every config/post-proc lever exhausted at the 0.330 ceiling, tested the
one direction the data pointed to: **shrink the scene, keep the splat budget.**
- Built `E:\vksplat_data\sharp_tight` NON-DESTRUCTIVELY: opened `sharp/sparse/0` read-only,
  selected the 30%-tightest spatial cluster of camera centers (pycolmap
  `projection_center()`, keep dist-to-median ≤ 30th pctile), copied ONLY those 174
  images_4 JPEGs to a new dataset dir, sliced a COPY of the reconstruction via
  `deregister_frame()` (this pycolmap uses rig/frame model — `frames.bin`/`rigs.bin`
  present; `deregister_image` does NOT exist, use `deregister_frame(im.frame_id)`).
  Result: 174 reg images, 49,015 pts3D, cam-bbox 13.6×5.5×5.6 → **4.3×3.4×3.3** (~10× less
  scene volume). Original `sharp/sparse/0` never written.
- **Trained Exp C config (1M, 50k, λ0.4) on sharp_tight → LPIPS-Alex 0.260** (VGG 0.314,
  PSNR 21.29, SSIM 0.700, 22 val). **21% better than the 0.330 full-scene ceiling — the
  ONLY lever in the entire session that beat 0.330, and it beat it big.**
- **Mechanism (confirms the lr_full analysis at memory.md:198):** achievable LPIPS ≈
  (splat budget) ÷ (scene volume) × (per-image sharpness). The 1M fog ceiling is fixed by
  hardware; the controllable term is SCENE VOLUME. Wide room = budget starved (0.33);
  tight scope = budget concentrated = real per-region detail (0.26).
- **CAVEAT (do not overclaim):** not a clean A/B — 22 tight-scene val views are intrinsically
  easier than 73 full-room views, so part of the 0.070 gain is "easier test." It proves the
  DIRECTION and magnitude decisively, not an exact equivalence to 0.330.
- **★ ACTIONABLE CAPTURE STRATEGY (supersedes the "only better data" vague conclusion):**
  to push toward LPIPS 0.20, do NOT add more wide-room images (lr_full 1321imgs=0.365,
  proven worse) and do NOT post-process (Wiener=0.662, fabricates fake detail). INSTEAD
  capture a SMALL, TIGHTLY-SCOPED scene — one focal area/corner, many well-distributed
  sharp angles of just that — so the fixed 1M-splat budget concentrates. Slicing alone hit
  0.26; a purpose-built tight capture should go further. Output:
  `E:\vksplat_output\sharp_tight_tight174_expC\`.

### 8b. 2026-05-19 LPIPS-0.200 push — splat frontier + post-proc re-confirmed EXHAUSTED
Goal was LPIPS-Alex 0.200 (vs 0.330 Exp C best). Ran the untried splat/post-proc levers
on sharp dataset (images_4 unless noted), Exp C config family, 73-img val. ALL failed to
beat 0.330; several regressed. Definitive table:
- **Wiener restore @ images_4** (deblur on already-sharp input): LPIPS-Alex **0.662** (vs
  0.330). Sharpening sharp frames injects ringing the splat faithfully learns. Dry-run
  var-of-Laplacian rose (+1861) — sharpness metric is a TRAP, it rewards halos. DEAD.
- **ssim_λ fine sweep** (only 0.2/0.4 known before): λ0.3→**0.345**, λ0.4→0.330,
  λ0.5→**0.333**. Clean single peak at 0.4. λ axis now fully mapped, exhausted.
- **B1: 80k steps, max_steps=80000** (LR-schedule confound fixed, unlike lr_full 100k/30k):
  LPIPS-Alex **0.364** — WORSE than 50k. Proves the 30k→50k gain was convergence, not
  "more steps better"; 50k is near-optimal, beyond it MCMC degrades. Settles the recurring
  "try 150k steps" idea: confirmed counterproductive, do NOT run long-step variants.
- **B2: cap_max 1.5M @ images_4** (only 1M known-good before): LPIPS-Alex **0.890**, PSNR
  14.19, used all 1.5M splats — **FOG**. ★ NEW: the 1M fog ceiling applies at LOW-res
  images_4 too, not just images_2. Catch-22 is resolution-independent. Falsifies the
  "more splats need more steps" hypothesis at the source: B2 fogged (not under-converged),
  and B1 already proved +steps degrade a healthy model — more steps on fog just churns fog.
- **Hi-res combo (NEW, closes the last gap): images_2 + cap500k + noise_lr 1e5 + λ0.4 +
  50k** — the one untested COMBINATION of all known fog-tamers at once. Result: PSNR 20.79,
  LPIPS-Alex **0.551**, 500k splats stable, NO fog (combo DID tame churn as Exp D predicted)
  — but still far worse than images_4@1M 0.330. Confirms memory.md:274/276 "fog-free
  images_2 is soft" with the full lever stack, not just isolated levers.
- **★ CONCLUSION: VkSplat hyperparameter + post-processing space is now EXHAUSTIVELY
  closed (splat count, steps, λ, hi-res combo, deblur all tested with zero gaps). 0.330 is
  the hard ceiling on this dataset/RX 7900 XTX. LPIPS 0.200 is NOT reachable by any
  config/post-proc — it requires better SOURCE DATA (sharper/denser capture).** COLMAP
  levers were investigated (degenerate sequential vs healthy exhaustive confirmed; current
  sparse/0 is the good 579/581, 125k-pt exhaustive run) but user opted not to run COLMAP
  experiments. `run_colmap.py` gained a harmless unused `--max-features` flag (default
  8192, no behavior change).

### 9. Blurry renders — ROOT CAUSE is blurry SOURCE data (May 2026)
- **Symptom**: full_dataset MCMC 1M/30k = PSNR 24.16, SSIM 0.841 (good numbers) but renders are visibly soft — no readable text, no fabric texture, no leaf detail.
- **Cause**: every source photo is heavily motion-blurred. Variance-of-Laplacian ≈ **15** across all 900 photos (sharp images score 500+; the single sharpest photo in the whole set scored 33). This is persistent **long-exposure motion blur in a dim room** (camera lengthens shutter in low light; any movement smears the frame).
- **Why PSNR was high anyway**: a blurry render matches blurry training images well. PSNR/SSIM are NOT reliable quality signals when source is uniformly blurred — judge by eye and LPIPS.
- **DEAD END — do not repeat**: THREE independent captures of this room all produced identical blur:
  - 900 photos: median var-of-Lap 15, max 33
  - 4K60 video #1 (`20260515_131428.mp4`, 8 min): **full scan of all 1971 sampled frames** → median 16, **max 117**, ZERO frames > 150 usable floor
  - 4K60 video #2 (`20260515_150023.mp4`): sample median 13, max 29
  - 4K30 video #3 (`20260515_152149.mp4`): sample median 15, room brightness 52/255 (very dark)
  - 4K24 video #4 (`20260515_154142.mp4`, MORE LIGHT): brightness 99/255, sample median 23, max 154 — **first real improvement**. BUT full scan (1926 frames): median 16, only 41 frames (2%) >100, **4 of 12 temporal segments** have usable frames — clustered where the user slowed down, NOT spread across the room. Not reconstructable (COLMAP needs sharp coverage everywhere).
  60 fps does NOT help because the limiting factor is per-frame *exposure time* (forced long by the dim room), not frame rate. Frame-selection cannot rescue a uniformly long-exposure capture — sharp frames only occur where the user pauses.
- **Confirmed levers (video #4 proved these)**: (1) MORE LIGHT directly raises sharpness — brightness 52→99 took max sharpness 30→167. (2) Sharp frames cluster at slow/paused moments. Therefore the fix is bright room + STOP-AND-SHOOT (or frequent ~2s pauses) so sharp frames occur throughout, not just at incidental stops. RAW vs JPEG is irrelevant — motion blur is baked into the sensor exposure before either format exists; only short exposure (light or fast manual shutter) + a still camera fixes it.
- **Only fix**: recapture with lots of light + stop-and-shoot (short exposure). See `CAPTURE_GUIDE.md`.
- **Diagnostic metric**: `cv2.Laplacian(gray, cv2.CV_64F).var()`. >300 sharp, 150–300 usable, <150 too blurry. Run `select_sharp_frames.py --check` on any new capture BEFORE COLMAP (saves ~88 min on bad data).
- **RESOLVED — what finally worked (`new fotos`, May 2026)**: 581 deliberate **stop-and-shoot** photos in a brighter room → median sharpness **386**, 91% usable, sharp across all 12/12 temporal segments (4000×1848 standard Samsung landscape, not 8K). Reconstruction shows real parquet pattern + crushed-velvet fabric texture + readable detail. THIS is the dataset to use: `E:\vksplat_data\sharp\`, output `E:\vksplat_output\sharp_mcmc1m\`.
- **Capture technique is THE decisive factor, not camera settings**: a separate "locked shutter + auto ISO" set of 1694 photos scored only median **70** (15% usable) — same camera trick but shot fast/while-moving. Deliberately stopping for each shot beats any exposure setting. Don't equate "locked shutter" with "sharp"; equate "camera physically still during exposure" with "sharp".
- **More photos ≠ better — REJECTED datasets (do not re-evaluate)**:
  - `E:\Downloads\new locked shutter auto iso` (1694 photos): median 70, 15% usable. Rejected.
  - `E:\Downloads\new fotos full dataset` (1380 photos, captured 20260516_0831-0902 — a DIFFERENT/longer session than the good `new fotos` 20260515_16xx): median 98, only 34% usable, blurry start+tail (segments 0,9-11 median 32-75). Even a coverage-spread bucketed-400 selection only reaches median 134 (below 150 floor) because ~1/3 of the walk-through has no sharp frames. Worse than `new fotos`. (User asked to try anyway — staged all 1380 → `E:\vksplat_data\full2\`, COLMAP exhaustive+primed running as empirical check.)
  - `E:\Downloads\bedroom 16-14` (1008 photos, NEW SCENE bedroom, 20260516_1419-1444): median **39**, only **1% usable** (13/1008), 4/12 segments covered. Brightness fine (101) — pure motion blur from walk-and-shoot (~1 photo/1.5s, too fast to settle). Rejected. **Confirms: adequate light is NOT enough; physically stopping for each shot is the decisive variable.** Bedroom is reconstructable with proper stop-and-shoot technique.
  - `E:\Downloads\bedroom 16-15 multifocus` (330 photos, 20260516_1500-1509): median **31**, **0.3% usable** (1/330), 1/12 segments, mixed portrait+landscape orientations. WORST yet. "Multifocus" attempt — varying focus does NOT fix motion blur (focus was never the problem). Same dead-end class as RAW/locked-shutter/60fps: tweaking a non-bottleneck camera setting. Rejected. Lesson reinforced: **no camera setting substitutes for holding the camera still during exposure.**
- **GOOD bedroom dataset — `E:\Downloads\bedroom normal camera`** (474 photos, 20260516_1522-1537): median **345**, **90% usable**, 59% >300, **12/12 segments covered**, brightness 126. Statistically equivalent to the `new fotos` LR winner (m386/91%). Folder name "normal camera" is the point — PLAIN default camera + stop-and-shoot discipline beat every gimmick (multifocus/locked-shutter/RAW/60fps all failed on the SAME room).
- **BEDROOM RECONSTRUCTION — SUCCESS (methodology generalizes)**: staged → `E:\vksplat_data\bedroom\`; COLMAP exhaustive+primed @12thr = **445/474 (94%) registered, 64K pts, correct intrinsics** (f=1349, k=0.036); MCMC 1M @ images_4 30k → **PSNR 20.46, SSIM 0.650, LPIPS-a 0.466**, healthy (`~110k relocate/1M`, no fog), 153s train. Render shows readable text (box labels, papers), tie-dye fabric texture, wood grain. Comparable to LR winner (PSNR 21.6). Output: `E:\vksplat_output\bedroom_mcmc1m\20260517_000645_bedroom\splat.ply`. **Proves: sharp capture + proven pipeline = good reconstruction, on a SECOND independent scene. The pipeline is reliable; capture discipline is the only variable that matters.**
- **WINNER stays**: `new fotos` (581 photos) → `E:\vksplat_data\sharp\` → `E:\vksplat_output\sharp_mcmc1m\20260515_211727_sharp\splat.ply`. No later/longer capture has beaten it. The yesterday session had stop-and-shoot discipline throughout; longer sessions lost it (sharp middle, rushed/blurry ends).
- **EMPIRICAL TEST of full2 (1380 photos, median 98) — ran the full pipeline anyway**: COLMAP registered **1361/1380 (98.6%), 162K points, correct intrinsics** — registration was EXCELLENT despite low sharpness. BUT training → PSNR **12.0**, LPIPS-alex 0.88, degenerate red/green fog (`~685k relocate/0 add`). Conclusively worse than new fotos (PSNR 21.6).
- **KEY LESSON — COLMAP success ≠ usable dataset**: COLMAP only needs repeatable SIFT keypoints for pose triangulation; blurry images still have those → high registration %. Gaussian Splatting fits splat colors directly to pixels → blurry pixels = fog. **The sharpness pre-check (`select_sharp_frames.py --check`, median>150) predicts RENDER quality, NOT COLMAP registration. Never judge a dataset by its COLMAP reg count — it looks deceptively good on blurry data.** 1380 blurry viewpoints did NOT average to sharp (motion smear differs per frame). The metric was right all along; it just predicts the right thing (training), not SfM.
- **Metric caveat confirmed**: sharp `new fotos` scored LOWER PSNR (21.57) and SSIM (0.682) than blurry full_dataset (24.16 / 0.841), but BETTER LPIPS (0.36 vs 0.42) and visibly far superior renders. On this kind of data, **judge by eye + LPIPS; ignore PSNR/SSIM** (blurry-vs-blurry inflates them).

### 10. COLMAP degenerate camera estimation — prime the intrinsics (May 2026)
- **Symptom**: on `sharp` dataset, COLMAP sequential matching registered only **16/581** images with garbage camera `f=2534, k=1.007` (k should be ~0.05). Data was confirmed sharp — not a data problem.
- **Cause**: bad initial pair → BA refined SIMPLE_RADIAL distortion into a degenerate basin (`ba_refine_extra_params=True` amplified it). Same class as Issue #4.
- **Fix**: `run_colmap.py` now has `--camera-params 'f,cx,cy,k'` (primes `reader_opts.camera_params`) and `--no-refine-extra`. Re-ran with **exhaustive matching** (proven on livingroom) + `--camera-params "1343.6,1000,462,0.051"` (known Samsung 2000×924 values from this file) → **579/581 registered, 125,280 points**, camera converged to f=1351, k=0.040 (correct). Exhaustive on 581 imgs ≈ 3.3 hr (O(N²), CPU-only) but robust.
- **Rule**: if COLMAP registers <50% or `k` is wildly off (>0.2 for SIMPLE_RADIAL on this phone), re-run exhaustive with `--camera-params` primed from the memory.md Camera Parameters table. Sequential matching is fast but fragile for stop-and-shoot (non-smooth path) captures — prefer exhaustive when registration fails.

### 11. Salvaging blurry captures you can't reshoot — `restore_frames.py` (2026-05-17)
- **Need**: capture conditions aren't always controllable and the location can be far/inaccessible — recapture (Issue #9's only documented fix) isn't always possible. Built a GPU deep-learning deblur salvage path so an imperfect dataset stays usable.
- **Tool**: `restore_frames.py`. Pretrained **NAFNet-GoPro-width64** (272 MB, `E:\vksplat_tools\models\NAFNet-GoPro-width64.pth`, HF mirror `nyanko7/nafnet-models`). Architecture is **self-contained in the script** (no `basicsr`/`timm` dependency — not installed, and `basicsr` is a ROCm risk). Verified: self-contained arch loads the official `ck['params']` (664 tensors) with **0 missing / 0 unexpected** keys. LayerNorm2d's custom autograd Function was replaced with a plain functional forward — identical math, same `weight`/`bias` params (checkpoint-compatible), inference-only so no backward (sidesteps the Issue #6 backward-kernel class entirely).
- **CRITICAL RULE — training tier ONLY, never `images_2`**: deblur output is not view-consistent (per-frame independent synthesis); feeding it to COLMAP corrupts SIFT matching/poses (Issue #1 sensitivity). Restore `images_4` only; poses are locked from clean `images_2` by training time. The script **refuses** if the target dir is an `images_2` tier.
- **GPU probe + auto-fallback**: `pick_device()` runs a real conv on the GPU inside try/except — ROCm here reports a torch build (`2.9.1+rocmsdk`, hip 7.2) but **fails at device enumeration** ("Failed to get device count", `cuda.is_available()==False`) whether the RX 7900 XTX is in or not. On probe failure it auto-falls-back to classical `unsharp` and never hard-blocks. `--dl-cpu` runs NAFNet on CPU (correct, ~0.02 img/s — validation only). `--cpu`/`--method unsharp|wiener|denoise-sharpen` = classical fallback.
- **Validated (DL-CPU, full_dataset blurry source)**: NAFNet end-to-end produces a genuine deblur — visual spot-check shows crisper parquet/wood-grain/leaf edges, **no ringing/hallucination/color shift** at `--dl-strength 0.7`. Classical `unsharp` raises var-of-Laplacian far more (+118 median) than NAFNet (+25) but that's metric inflation (unsharp boosts HF noise); judge DL by eye + the deferred A/B, not Laplacian delta.
- **Guardrail gotcha (fixed)**: first guardrail used absolute `clip_frac > 0.05` and rejected 3/4 frames — wrong, because these dim-room captures *already* have 4–19% pixels ≥254 (bright windows/lamps). Corrected to trigger on the **increase** in clipping caused by restoration (`clip1 - clip0 > 0.03`), plus `s1 < s0`. After fix: 0/4 rejected.
- **GPU PATH RESOLVED — runs on the GTX 1070 Ti via the NVIDIA env (2026-05-17)**: the default `python` has the ROCm torch (probe fails → CPU fallback). The pre-existing **`E:\nvidia-gsplat\python312\python.exe`** has **torch 2.5.1+cu121** and sees the **1070 Ti** (`cuda_avail True`, device count 1). Installed `opencv-python-headless` + `scipy` into it (the env was gsplat-minimal: had torch/numpy/PIL, lacked cv2/scipy; headless avoids Qt). `pick_device()` → `cuda | NVIDIA GeForce GTX 1070 Ti`; dry-run on full_dataset = "GPU ready", real NAFNet inference, same deltas as the CPU run (1→281, 6→96) confirming correctness, ~100× faster than `--dl-cpu`. **Key: NAFNet is inference-only so Pascal sm_61 is fine — it does NOT hit the Volta-only backward kernels that make gsplat a dead-end on this card (Issue #6 / gsplat verdict).** Added `sys.path.insert(0, scriptdir)` to `restore_frames.py` so sibling imports resolve from any cwd/env. Run command: `E:\nvidia-gsplat\python312\python.exe E:\vksplat\restore_frames.py <tier> --method dl`.
- **A/B training proof — RESOLVED (2026-05-18)**: Kitchen A/B complete. Run B renders vs blurry GT scored **BETTER** LPIPS than Run A baseline (Alex 0.427 < 0.441, VGG 0.436 < 0.444). Deblur does NOT hurt 3D reconstruction; it's neutral-to-slightly-positive perceptually. Per plan decision rule: deblur path is **validated** — safe to apply to other datasets. Rows 27–29 in results table. eval.json paths: Run A `E:\vksplat_output\kitchen_ab_clean\20260518_170813_kitchen\eval.json`, Run B `E:\vksplat_output\kitchen_ab_restored\20260518_171134_kitchen\eval.json`.
- **Standing rule**: this is a salvage tool, not a substitute for sharp capture (Issue #9 / CAPTURE_GUIDE remain primary). Use it only when reshoot is genuinely impossible.

### 12. NAFNet rainbow-block artifacts — content-specific divergence, fixed by TILE-level fallback (2026-05-18)
- **Symptom**: restored images had rectangular blocks of rainbow/RGB static (looked like a corrupt JPEG region), intermittent across frames, ~tile-sized. Present on kitchen + bedroom outputs.
- **WRONG first diagnosis**: assumed GPU fp32 NaN instability on the 1070 Ti. Added `np.isfinite` guard. **Did not fix it** — artifact persisted on the 7900 XTX too.
- **Real root cause**: NAFNet-GoPro emits **finite but wildly unbounded** output on *specific image content* — measured tile output range **−8001..+7959** (valid is ~[0,1]); `np.clip(0,1)` then crushes that chaos into 0/1 noise → rainbow block. `isfinite` never catches it because the values are finite. **Confirmed content-specific, NOT environment**: same tile gives identical explosion on CPU fp32, GPU fp32, AND GPU fp64; the same content is clean at a different tile size — i.e. it's NAFNet's architecture having no output bound, triggered by certain patches (dark glossy/specular surfaces, saturated fine-texture fabric). Input preprocessing is CORRECT (verified vs official `basicsr/demo.py`: RGB, float32, /255 → [0,1]).
- **FIX (the working one)**: two-level guard in `restore_frames.py`:
  1. **Tile level** (`_infer_tiled`): a tile whose output is non-finite OR has >2% pixels outside [−0.3,1.3] is declared diverged and replaced with that tile's **input pixels** (clean, just un-deblurred). Counted in `_BAD_TILES`. The feather-blend makes the swap seamless — no visible seam between deblurred and fallback regions.
  2. **Frame level** (main loop): do **NOT** discard the whole frame just because some tiles diverged (the earlier `nbad>0 → keep original` was wrong — it threw away 52/85 kitchen frames that were 80%+ validly deblurred). Only the genuine net-quality guardrail remains: keep original iff `s1<s0` or clipping *increased* >0.03.
- **Result**: kitchen worst-case frames (incl. the window-glass one that had the rainbow block) now come out **deblurred AND artifact-free**, seams invisible, ~1/4 frames kept-original by the real guardrail (net-worse) vs 52/85 before. Verified visually on multiple frames.
- **Lesson**: clip/clamp do not sanitize a model that has no output bound — you must detect the out-of-range explosion and substitute, and do it at the smallest unit (tile) so one bad region doesn't waste the whole frame's deblur. Divergence is per-content and unavoidable with this checkpoint; the tool degrades gracefully instead.
- **SUPERSEDES Issue #11's "use the NVIDIA env" instructions**: 2026-05-18 the 7900 XTX is reinstalled, 1070 Ti out, ROCm torch sees the XTX — run `restore_frames.py` / `restore_all.py` with the **default `python`** (see updated System Specs + 🐍 section). `restore_all.py` is the batch orchestrator (single-instance lock; 4 input classes; full-res→downscale; A/B-vs-baseline still deferred to VkSplat training).

### 13. video1 (3852-frame portrait 8K video) — blur salvage attempt (2026-05-19)
- **Dataset**: `E:\vksplat_data\video1\`, 3852 frames from a portrait 8K video. Source `images_2` median var-of-Laplacian sharpness **47** (only **439/3852 = 11%** clear the 150 floor) — **the blurriest dataset attempted** (worse than the rejected full2 median-98, bedroom-16-14 median-39 class).
- **Raw `images_2` COLMAP: 3/3852 (0.08%) registered — FAILED.** Garbage camera `f=6909, k=-2.0`. The stale `sparse/0/` (mtime 00:54, 3 reg / 430 pts) is this failed run's output. Confirms Issue #9: motion blur destroys SIFT pose repeatability across views — restoration is **mandatory** before COLMAP on this data.
- **Restorations staged (3852 imgs each)**: `images_4` NAFNet (→`images_4_restored`), Restormer (`images_4_restormer`), Wiener (`images_4_wiener`), Unsharp (`images_4_unsharp`); `images_2` Wiener (`images_2_wiener`), Unsharp (`images_2_unsharp`). All complete.
- **`images_2_unsharp` COLMAP (sequential, overlap 10, PID 17304)**: extraction + matching **completed and healthy** — DB has 3852 keypoints/descriptors, 14336 pairs, **8895 with valid two-view geometry** (vs the raw run's near-total match failure: unsharp restored SIFT repeatability). BUT the process **died/was-killed during incremental mapping** (no log redirect; no fresh `sparse/`; PID gone). The 1.02 GB `database.db` (mtime 09:09) holds the unsharp features+matches — **mapping can be resumed from the populated DB without re-extracting/re-matching** (hours saved). `images_2_wiener` COLMAP still QUEUED.
- **BAD-Gaussians evaluated → NOT VIABLE here.** Needs NVIDIA CUDA 11.8 + tiny-cuda-nn + gsplat + nerfstudio 1.0.3 (`python<3.11`). Only the AMD 7900 XTX is installed; gsplat already dead-ended twice (the "version vise", see FINAL VERDICT); tiny-cuda-nn has no ROCm port. Same hardware-floor blocker class as the gsplat verdict — **do not retry without a ≥sm_70 NVIDIA GPU.**
- **All dedicated motion-blur-3DGS methods are NVIDIA-only**: BAD-Gaussians, DeblurGS (2404.11358), BSGS (2510.12493), GS-on-the-Move (2403.13327), Robust-GS (2404.04211), Deblurring-3DGS — every one builds on the CUDA 3DGS rasterizer or nerfstudio+gsplat. The blur model lives **inside the differentiable rasterizer** (per-pixel pose integration / virtual sub-views), so it **cannot be ported onto VkSplat** (Slang/Vulkan, no pose-opt or blur-convolution hook).
- **★ AMD-viable conclusion**: the in-flight pipeline (**image restore → COLMAP on restored → VkSplat train**) IS the correct and only deblur strategy on this hardware — there is no better engine to switch to. Judge `video1` purely by whether a restored variant gets COLMAP to register well **AND** VkSplat renders non-fog (Issue #9: high COLMAP reg-% on blurry data still trains to fog; median-47 is far below the new-fotos median-386 winner — treat as a salvage stress test, not a quality run). Cascade-restoration experiments (Wiener→Unsharp, Unsharp/Wiener→NAFNet, NAFNet/Restormer→Unsharp, strength sweeps) being screened via `restore_frames.py --dry-run` to try to clear the floor.
- **★★ KEY DIAGNOSTIC (2026-05-19): the unsharp COLMAP failure was NOT a blur/SIFT problem — it was the degenerate-k camera divergence (Issue #10), and PRIMING FIXES IT.** `images_2_unsharp` mapping resumed from the populated DB with **no camera prior** registered only **2/3852** (`f=5135, k=4.09` — absurd) and bailed in 2.5s. Re-run with `--camera-params "1400,1020,574,0.0" --no-refine-extra` (SIMPLE_RADIAL primed, k locked at 0) → COLMAP **registers steadily, 500+ frames and climbing**, each new image seeing 800–1200 triangulated points (healthy). So the bottleneck on `video1` was COLMAP *configuration* (unconstrained focal/distortion → degenerate basin), **not restoration quality**. Implication: the existing single-stage restorations may already be sufficient — the cascade experiments are now a *secondary* lever, not the critical path.
- **Tooling added**: `colmap_map_only.py` — resumes `pycolmap.incremental_mapping` from an already-populated `database.db` (skips re-extraction/re-matching, hours saved when a run dies in mapping). Supports `--camera-params 'f,cx,cy,k'` (edits the SQLite `cameras` table directly — `pycolmap.Database` is abstract/uninstantiable in this build, so the documented pycolmap-API priming does NOT work here; raw sqlite3 UPDATE of `model`/`params` blob/`prior_focal_length` is the working method), `--no-refine-extra`, `--no-refine-focal`. When priming, it works on a per-run COPY of the DB (in the output dir) so the expensive source DB stays pristine for other priors. Portrait-8K-video `video1` is **2040×1148** (NOT the landscape 2000×924 of old photo datasets — the memory Camera Parameters table f=1343.6 does not apply; primed f=1400 @ k=0 works).
- **★ COLMAP global BA is VERY SLOW (not hung) at ~1040 registered cameras on this CPU/16GB — 2026-05-19.** BOTH primed runs (unsharp p1400 @ reg=1042, wiener @ reg=1038) went log-silent for 20–46 min right after triggering "Retriangulation and Global bundle adjustment". Initially misread as a hang. **Verified via `Win32_PerfFormattedData...PercentProcessorTime`: the wiener python proc was at CPU%=358 (≈3.6 cores) during the silence — actively crunching, NOT deadlocked.** pycolmap's periodic global BA is a single Ceres solve over ~1000+ poses + a huge noisy tie-point cloud (blurry video → many spurious 3D points); on CPU-only + 16 GB it legitimately takes 20–40+ min and emits ZERO incremental log. **Rule: do NOT judge a COLMAP run dead from log silence alone after a "Global bundle adjustment" line — check `PercentProcessorTime`; >100% = working, wait it out. Budget hours, not minutes, for 3852-frame video COLMAP on this box.**
- p1400 (unsharp) ALSO had a real second problem: a duplicate `colmap_map_only.py` was spawned (harness said task "timed out" but the python proc was still alive; relaunch → TWO procs on the SAME `sparse_unsharp_p1400/database.db` → added SQLite contention). Killed both (PIDs 16484, 2808). **Tooling rule: `colmap_map_only.py` needs a UNIQUE per-run out dir AND a single-instance guard — verify via `wmic process ... get CommandLine` before relaunch; the harness "timeout" is NOT proof the proc died.** Unsharp abandoned regardless (weakest restoration: sample median 140, 8/24 >150). **Wiener** (median 392, 24/24 >150) is the real candidate — full primed sequential COLMAP on `images_2_wiener` with its own `database_wiener.db` + `sparse_wiener/` (single clean proc, no contention; the slow-BA wait is expected, not a fault).
- **★★ WIENER COLMAP — SUCCESS (2026-05-19 23:45):** `images_2_wiener` primed sequential COLMAP completed. **Model 0: 3850/3852 registered (99.95%), 312,770 3D points.** Camera converged sanely: SIMPLE_RADIAL 2040×1148, **f=1302.65** (from primed 1400, ~7% refinement), cx=1020, cy=574, **k=0.0 (locked)** — no degenerate basin. Total wall time **49,579 s ≈ 13.8 h** (CPU-only; slow-global-BA cycles dominate — see "slow BA not hung" note). Output: `E:\vksplat_data\video1\sparse_wiener\0\` (cameras/images/points3D.bin, ~166 MB). **Highest registration rate this project has achieved (99.95% vs 99.8% `lr_full` prior best), proving the Wiener-restoration + camera-priming recipe salvages the median-47 `video1` data that raw/unsharp COLMAP failed on** (raw 0.08%, unsharp 0.05% pre-priming, 2/3852 unsharp post-priming-with-dup-procs). Next: stage `sparse_wiener/0` → `sparse/0` and VkSplat smoke-train on `images_4_wiener` (Exp-C cfg: MCMC 1M, ssim_λ 0.4, 50k steps); judge by eye + LPIPS, never PSNR (Issue #9 caveat — high COLMAP reg-% on blurry source can still train to fog; median-47 << new-fotos median-386, so temper expectations and ENABLE Wiener-restored training tier `images_4_wiener`, not raw `images_4`).
- See project-memory note `bad-gaussians-not-viable.md` (incl. ZLUDA rejected).

---

## Known Issues / Limitations

### RAM Constraint (16 GB)
- Loading 246 images at 4000×1848 = ~5.5 GB RAM just for pixels
- Plus Python, COLMAP data, OS → can push past 16 GB
- **Full-resolution training with all 246 images may crash the PC**
- Workaround: use fewer images at full-res, or train at half-res

### GPU Hang at 3M Splats + Full Resolution
- Training froze at step 11498 when splat count hit 3,000,000 at 4000×1848
- Process alive but zero CPU/GPU activity (28 MB working set)
- Likely Vulkan compute shader hang due to sorting buffer size
- **cap_max=2M or lower recommended at full resolution**

### Full-Res Training with All 246 Images Crashes PC (16 GB RAM)
- 246 images × 4000×1848 × 3 bytes ≈ 5.5 GB just for pixels
- Plus Python/OS overhead pushes past 16 GB
- **Workaround**: use a subset (every 3rd image = 83 images, ~1.8 GB pixels)

### Full-Res with Fewer Images = Worse Quality
- 83 images at 4000×1848 with 2M splats → PSNR 18.39
- 246 images at 2000×924 with 1M splats → PSNR 24.38
- **More images at lower resolution beats fewer images at higher resolution**
- The number of training viewpoints matters more than per-pixel detail

### VkSplat Quality Ceiling
- PSNR 24.38 with 249 photos at 2000×924 is best so far
- Doubling cap_max (1M→2M) didn't improve quality on kitchen dataset
- May need higher-resolution training, better camera model, or different 3DGS implementation

### Video Frames Not Useful
- Adding 485 video frames (1fps from 8K videos) to 249 photos made quality WORSE
- Video codec compression + motion blur + orientation issues
- Video frames from portrait/landscape mix with different lens modes

---

## Directory Structure

```
E:\vksplat\                     # VkSplat source + training scripts
  vksplat\                      # C++/Python source (pip install -e .)
  train_livingroom.py           # General training entry (now takes --dataset-dir / --output-base)
  train_kitchen.py              # Kitchen photos training (hardcoded path)
  train_combined.py             # Combined photos+video training (hardcoded path)
  train_photos.py               # Original phone photos training (hardcoded path)
  run_colmap.py                 # COLMAP SfM automation (exhaustive matching)
  run_colmap_sequential.py      # COLMAP with sequential matching (for timestamped photos)
  prepare_dataset.py            # EXIF-strip + orient + resize raw photos → tiered dataset layout
  select_sharp_frames.py        # video/photo → sharpest-frame selection (bucketed) → tiered layout; --check scores only
  CAPTURE_GUIDE.md              # how to capture a SHARP dataset (anti-motion-blur)
  run_eval.py                   # Subprocess evaluation script
  filter_recon.py               # Remove outlier cameras from COLMAP
  check_outliers.py             # Analyze camera position outliers

E:\vksplat_data\
  livingroom\                   # 249 living room photos (Samsung phone, 4000×1848 source)
    images\                     # Full res (4000×1848)
    images_2\                   # Half res (2000×924)
    images_4\                   # Quarter res (1000×462)
    sparse\0\                   # COLMAP output (for images_2)
    sparse\0_fullres\           # COLMAP with 2× scaled intrinsics
  livingroom_combined\          # Photos + video frames
    photos\                     # 249 photos at 2000×924
    video1\                     # 230 frames from portrait 8K video
    video2\                     # 255 frames from landscape 8K video
    sparse\1_clean\             # Filtered COLMAP (720 images)
  kitchen\                      # 85 kitchen photos
    images_2\, sparse\0\
  full_dataset\                 # 900 sequential 8K portrait photos (different camera/zoom — see below)
    images_2\                   # 2040×4418 (EXIF-rotated portrait)
    images_4\                   # 1020×2209
    sparse\0\                   # COLMAP sequential matching, 891/900 registered, 216k points

E:\vksplat_output\              # Training outputs (PLY + renders + eval.json)
```

## Camera Parameters (Samsung Phone — old datasets)

| Resolution | Focal Length | Principal Point | Distortion (k) |
|-----------|-------------|----------------|-----------------|
| 2000×924 (images_2) | 1343.6 | (1000, 462) | 0.051 |
| 4000×1848 (images) | 2687.1 | (2000, 924) | 0.051 |

Focal/width ratio ≈ **0.67** → wide-angle phone lens. EXIF orientation = 6 on phone source.

## Camera Parameters (`full_dataset` — different camera/zoom)

| Resolution | Focal Length | Principal Point | Distortion (k) |
|-----------|-------------|----------------|-----------------|
| 2040×4418 (images_2, portrait) | 2948.1 | (1020, 2209) | 0.040 |
| 1020×2209 (images_4, portrait) | 1474.0 (scaled by loader) | (510, 1104.5) | 0.040 |

Focal/width ratio ≈ **1.44** — roughly 2× higher than the Samsung wide-angle. Likely a different camera, telephoto lens, or zoomed capture. Source images are 8160×3768 with EXIF orientation 6 (rotates to 3768×8160 portrait). The known-good Samsung params do NOT transfer; let COLMAP estimate from scratch.

---

## Useful Commands

```bash
# Stage a flat folder of raw photos into the tiered dataset layout (EXIF-aware)
python E:\vksplat\prepare_dataset.py "E:\Downloads\50mb full dataset" E:\vksplat_data\full_dataset --skip-fullres
# default tiers: images_2 width 2040, images_4 width 1020 (override with --width-2 / --width-4)

# Run COLMAP — exhaustive matching (best for unordered photos)
python E:\vksplat\run_colmap.py E:\vksplat_data\livingroom --image-subdir images_2

# Run COLMAP — sequential matching (for timestamped / sequentially captured photos)
python E:\vksplat\run_colmap_sequential.py E:\vksplat_data\full_dataset --image-subdir images_2

# Train MCMC 1M at half-res on the livingroom dataset (original use)
python E:\vksplat\train_livingroom.py --strategy mcmc --cap-max 1000000 --steps 30000 --tag lr_mcmc1m

# Train MCMC on a NEW dataset (using the new --dataset-dir flag)
python E:\vksplat\train_livingroom.py --dataset-dir E:\vksplat_data\full_dataset --image-dir images_4 --strategy mcmc --cap-max 1000000 --steps 30000 --tag full_mcmc1m

# Train at full-res (needs sparse/0_fullres)
python E:\vksplat\train_livingroom.py --image-dir images --strategy mcmc --cap-max 2000000 --steps 50000 --refine-stop-iter 30000 --tag fullres_mcmc2m

# Evaluate a training output
python E:\vksplat\run_eval.py E:\vksplat_output\<output_dir>
```

---

## `full_dataset` (May 2026 session) — what worked

**Source**: `E:\Downloads\50mb full dataset` — 900 sequential timestamped JPEGs (`20260514_HHMMSS.jpg`), 8160×3768 source resolution, EXIF orientation 6, ~9.3 GB total. Not the documented Samsung phone (different focal length profile).

**Pipeline that worked**:
1. **Staging** (`prepare_dataset.py`, ~13 min): EXIF-aware load → manual orientation 6 (rotate 270° CW so saved pixels are upright portrait) → resize to `images_2` (2040×4418) and `images_4` (1020×2209) → JPEG q95 with EXIF stripped. Skipped `images/` (full-res) — pointless given GPU hang at >3M splats + 4K-wide.
2. **COLMAP** (`run_colmap_sequential.py`, ~88 min total): CPU SIFT (`first_octave=0`, 4 threads) → sequential matching (`overlap=10`, `quadratic_overlap=True`) → incremental mapping. **891/900 registered, 216,580 3D points**, `f=2948 at 2040-wide`, `k=0.040`. Mapping was the bottleneck (33 min).
3. **Smoke training** (40.5 s on 7900 XTX): `train_livingroom.py --dataset-dir ... --image-dir images_4 --strategy mcmc --cap-max 500000 --steps 5000 --skip-eval`. 500k splats, 671 MB peak VRAM, 779 train / 112 val. Visual sanity-check vs source: structure correct, no EXIF rotation bug.

**Lessons specific to this dataset**:
- 1800-file count from naive case-insensitive glob on Windows (double-matches `*.jpg` and `*.JPG`); actual count is 900. Always `len(set(files))` or `os.listdir` then filter by extension.
- 8K-source `images_2` is too large at 2040×4418 portrait for 900 images on 16 GB RAM (~24 GB pixels uncompressed). **`images_4` is the practical training tier for this dataset**, not `images_2`.
- Sequential matching cut COLMAP matching from "hours" (estimated for exhaustive on 900) to ~22 minutes.
- pycolmap GPU SIFT not available on this install (no CUDA/OpenGL support compiled in) — and irrelevant since the GPU is AMD anyway.

## Next Steps to Try
- Full run on `full_dataset`: `train_livingroom.py --dataset-dir E:\vksplat_data\full_dataset --image-dir images_4 --strategy mcmc --cap-max 1000000 --steps 30000 --tag full_mcmc1m` (with eval)
- If full_dataset PSNR caps below ~24, try `images_2` (2040×4418) with a 450-image subset to fit RAM
- Full-res training (livingroom) with fewer images (e.g., every 3rd image) to stay under 16 GB RAM
- `cap_max=2M` at full resolution to see if more splats + higher res helps
- Progressive training: start at images_4, then images_2, then images
- Better camera model: OPENCV (8 params) instead of SIMPLE_RADIAL (if enough images)
- Consider enabling COLMAP loop detection (`--loop-detection`, requires vocab tree) if a sequential dataset has revisits — currently `quadratic_overlap=True` handles short-range loops only

---

## gsplat + ROCm via WSL2 (In Progress)

### Why Switch from VkSplat
- VkSplat quality ceiling: PSNR ~24-25 max with MCMC strategy
- VkSplat shared memory limitation on RDNA3 (32KB vs 43KB needed for optimal backward pass)
- gsplat is the reference implementation used by nerfstudio, more features (appearance embeddings, etc.)

### Setup Requirements
- **BIOS**: AMD SVM (Secure Virtual Machine) must be enabled
- **Windows Feature**: Virtual Machine Platform must be enabled
- **WSL2**: Ubuntu 22.04 (recommended for ROCm compatibility)
- **ROCm 7.2.1**: Latest version with ROCDXG production support for WSL2
- **ROCDXG**: User-mode library bridging ROCm to Windows GPU via DXCore (/dev/dxg)
- **GPU Target**: gfx1100 (RDNA3 = RX 7900 XTX)
- **HSA_OVERRIDE_GFX_VERSION=11.0.0**: Required for consumer RDNA3 GPUs

### Key Findings from Research
1. **RX 7900 XTX IS officially supported** for ROCm on WSL2 (as of Adrenalin 26.2.2)
2. AMD's ROCm/gsplat fork (v1.5.3b2) officially targets MI300X/MI325X (gfx942)
3. gsplat can be built from source targeting gfx1100 with `PYTORCH_ROCM_ARCH=gfx1100`
4. nerfstudio's splatfacto method uses gsplat as backend
5. PyTorch ROCm is available via `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`

### Blockers Encountered — ALL RESOLVED 2026-05-17, GPU NOW VISIBLE IN WSL
- ~~Virtualization disabled in BIOS~~ — NOW ENABLED (verified: VirtualizationFirmwareEnabled=True, HypervisorPresent=True, wsl.exe works).
- ~~ROCm can't see GPU (`hsa_init Failed`, no /dev/kfd)~~ — **SOLVED via librocdxg**.
  - Root cause was NOT a ROCm/Adrenalin version mismatch (AMD docs: ROCDXG is
    version-flexible, unlike legacy roc4wsl). Real cause: the `librocdxg`
    Linux↔Windows-GPU translation layer was simply never installed.
  - `/usr/lib/wsl/lib/` only has NVIDIA stub libs — those are harmless WSL
    defaults, NOT the blocker. ROCDXG talks to the AMD driver via DXCore
    (/dev/dxg), bypassing /dev/kfd entirely (WSL has no /dev/kfd, that's normal).
- **WORKING SETUP (verified 2026-05-17, RX 7900 XTX detected as gfx1100):**
  1. WSL2 Ubuntu 22.04 distro `ubuntu-rocm`, VHDX on `F:\wsl\ubuntu-rocm\` (NOT C:)
  2. ROCm 7.2.1 via `amdgpu-install -y --usecase=rocm --no-dkms`
     (the `wsl` usecase does NOT exist in the 7.2.1 jammy installer — use
     `rocm --no-dkms`; WSL needs no kernel driver)
  3. Build `librocdxg` 1.2.0 from github.com/ROCm/librocdxg:
     `cmake .. -DWIN_SDK="/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared"`
     then make / make install → `/opt/rocm-7.2.1/lib/librocdxg.so`
  4. Verify: `HSA_ENABLE_DXG_DETECTION=1 rocminfo` → shows
     `Name: gfx1100 / Marketing Name: AMD Radeon RX 7900 XTX`. **rocminfo exit=0.**
  - Adrenalin 26.5.2 (current) is sufficient; NO Windows driver change needed.
  - `HSA_ENABLE_DXG_DETECTION=1` env var is REQUIRED at runtime for GPU detection.
- Build scripts: `\\wsl$\ubuntu-rocm\root\{install_rocm2,build_rocdxg}.sh`.

### PyTorch-ROCm on WSL: the bundled-HSA trap (diagnosed 2026-05-17)
- **Symptom**: `rocminfo` sees gfx1100 fine, but `torch.cuda.is_available()=False`,
  `device_count=0`. HIP verbose log: `Initializing HSA stack` →
  `hipGetDeviceCount: Returned hipErrorNoDevice`.
- **ROOT CAUSE**: `pip install torch ... rocm6.3` bundles its OWN
  `libhsa-runtime64.so` + `libamdhip64.so` (ROCm 6.3) in
  `venv/.../torch/lib/`. That legacy 6.3 HSA runtime has NO DXG/WSL support, so
  it can't reach the GPU through `/dev/dxg`+librocdxg. rocminfo works because it
  uses the SYSTEM ROCm 7.2.1 HSA (DXG-aware); torch fails because it uses its
  bundled 6.3 HSA. `HSA_OVERRIDE_GFX_VERSION=11.0.0` does NOT help (not a
  gfx-recognition issue — the bundled runtime can't see /dev/dxg at all).
- **FIX**: replace torch's bundled HSA/HIP libs with the system ROCm 7.2.1 ones
  (symlink `torch/lib/libhsa-runtime64.so` → `/opt/rocm/lib/...`, same for
  libamdhip64), OR LD_PRELOAD the system libs. Then torch uses the DXG-aware
  runtime. (No torch wheel built against ROCm 7.x exists yet; 6.3 is newest.)
- NOTE: torch wheel got corrupted (`librocsparse.so cannot read file data`)
  when the PC rebooted mid-`pip install`; fixed by clean `--no-cache-dir`
  reinstall. Reboots mid-install = corrupt wheels; always verify ELF integrity.

### ✅ PyTorch-ROCm WORKING on RX 7900 XTX / WSL (verified 2026-05-17)
- `torch 2.9.1+rocm6.3`, `torch.cuda.is_available()=True`, device 0 =
  "AMD Radeon RX 7900 XTX", real 2048² GPU matmul ran in 0.3s.
- **THE FIX (symlink torch's bundled HSA/HIP → system ROCm 7.2.1)**:
  ```
  TLIB=/root/venv/lib/python3.10/site-packages/torch/lib
  mkdir -p $TLIB/_bundled_backup
  mv $TLIB/libhsa-runtime64.so* $TLIB/libamdhip64.so* $TLIB/_bundled_backup/
  ln -sf /opt/rocm/lib/libhsa-runtime64.so.1 $TLIB/libhsa-runtime64.so
  ln -sf /opt/rocm/lib/libhsa-runtime64.so.1 $TLIB/libhsa-runtime64.so.1
  ln -sf /opt/rocm/lib/libamdhip64.so.7.2.70201 $TLIB/libamdhip64.so
  ln -sf /opt/rocm/lib/libamdhip64.so.7.2.70201 $TLIB/libamdhip64.so.6
  ```
  then run with `HSA_ENABLE_DXG_DETECTION=1`.
- `LD_LIBRARY_PATH=/opt/rocm/lib:...` alone did NOT work (torch dlopens its
  bundled libs by path) — the symlink replacement is required.
- Benign noise (does NOT block compute): `[GetSegmentId] Failed to get segment
  id for type 1` printed at init — matmul still succeeds.
- venv: `/root/venv` in distro `ubuntu-rocm` (vhdx on F:). Activate +
  `export HSA_ENABLE_DXG_DETECTION=1` before any torch GPU use.
- Fix script: `\\wsl$\ubuntu-rocm\root\fix_torch_hsa.sh` (idempotent).

### gsplat on gfx1100: prebuilt wheel CRASHES (2026-05-17)
- `pip install amd_gsplat --extra-index-url pypi.amd.com/rocm-7.0.0` → installs
  `amd_gsplat 1.5.3`, imports fine, `torch.cuda` sees the 7900 XTX.
- BUT real `gsplat.rasterization(...)` on `cuda` → **Segmentation fault (core
  dumped)** in native HIP kernel (NOT a catchable Python exception). Preceded by
  `[GetSegmentId] Failed to get segment id for type 1`.
- ROOT CAUSE: the prebuilt `amd_gsplat` wheel's HIP kernels are compiled for
  datacenter **gfx942** (MI300X/MI325X — AMD's only officially supported gsplat
  targets). They install+import on consumer **gfx1100** but crash on kernel
  dispatch (arch mismatch / no valid kernel image for gfx1100).
- NEXT: must build gsplat FROM SOURCE with `PYTORCH_ROCM_ARCH=gfx1100` so HIP
  kernels are compiled for RDNA3. (The earlier install script's source fallback
  never ran because the wheel imported OK — masking that its kernels are
  wrong-arch. "imports fine" ≠ "kernels run".)
- STATUS: PyTorch-ROCm itself works on gfx1100 (matmul OK). The open question is
  whether gsplat's HIP kernels compile+run on gfx1100 from source.
- **AMD `amd_gsplat` sdist is a STUB (verified 2026-05-17)**: `pip download
  --no-binary :all: amd_gsplat` yields a source pkg that builds a 1.3 KB wheel
  in <1s and installs NOTHING importable (`ModuleNotFoundError: gsplat`). AMD
  ships ONLY a gfx942 binary wheel + a placeholder sdist — there is NO buildable
  AMD gsplat source on pypi.amd.com. So gfx1100 cannot be reached via AMD's pkg.
- Upstream `nerfstudio-project/gsplat` build → also failed (its kernels are
  CUDA, hipify-on-RDNA3 incomplete). Not the right source anyway.
- **CORRECT source = `github.com/ROCm/gsplat`** (AMD's HIP-ported fork; the
  prebuilt `amd_gsplat` wheel is built FROM this, but for gfx942). Building it
  for gfx1100 also failed at HIP kernel compile.
- **★ ROOT CAUSE of ALL gsplat build failures (found 2026-05-17): ROCm VERSION.**
  ROCm/gsplat issue #12 = open feature request "Please add support for ROCm
  7.2". Official docs require **ROCm 6.4.3 or 7.0.0/7.0.x**. We installed ROCm
  **7.2.1** — TOO NEW. gsplat 1.5.3b2 HIP kernels reference ROCm APIs/headers
  changed in 7.2 → "Error compiling objects for extension". NOT a gfx1100
  incompatibility — a ROCm-runtime-too-new mismatch.
- **FIX IN PROGRESS**: downgrade WSL ROCm 7.2.1 → 7.0.2 (available at
  repo.radeon.com/amdgpu-install/7.0.2/), rebuild librocdxg + re-point torch
  HSA/HIP symlinks to 7.0.x, then rebuild ROCm/gsplat against supported 7.0.x.
  All WSL-side (F: vhdx), zero Windows/C: impact. amdgpu-install dirs available:
  6.4.3, 7.0, 7.0.1, 7.0.2, 7.2, 7.2.1.
  - 7.0.2 install gotchas (all WSL/apt, not GPU): needs `--allow-downgrades`;
    needs `-o Dpkg::Options::=--force-confnew` (conffile prompt EOFs in
    background); `amdgpu-install --usecase=rocm` demands
    `hsa-runtime-rocr4wsl-amdgpu` which is ABSENT in 7.0.2 repo (legacy roc4wsl,
    superseded by librocdxg) → install core libs DIRECTLY instead:
    `apt install rocm-hip-runtime rocm-hip-libraries rocm-llvm hipcc rocminfo
    rocm-core hip-runtime-amd` (all present as 7.0.2 pkgs; verified via
    apt-cache search). Then rebuild librocdxg against 7.0.2.
- **DOCKER path REJECTED (checked rocm.docs.amd.com/projects/gsplat 2026-05-17)**:
  AMD's gsplat docker image `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0`
  requires `--device=/dev/kfd` — bare-metal Linux ROCm device. **WSL2 has NO
  /dev/kfd** (uses /dev/dxg+librocdxg). The image is built for bare-metal
  datacenter (MI300X/gfx942), not WSL2 + consumer gfx1100. Docker fixes the
  version-mismatch but reintroduces the WSL-incompatibility we solved with
  librocdxg, AND keeps the gfx942 kernel-arch risk. Not viable without
  bare-metal Linux (dual-boot, declined for disk reasons). Native WSL path is
  the only route.

### ★★ FINAL VERDICT: gsplat-on-ROCm NOT VIABLE here (2026-05-17) ★★
- **The version vise (unsolvable with current AMD stack):**
  - ROCm **7.2.1**: librocdxg works, gfx1100 detected, PyTorch GPU matmul OK —
    but ROCm/gsplat **won't build** (7.2 unsupported, issue #12; HIP kernel
    compile fails).
  - ROCm **7.0.2** (gsplat-supported): librocdxg 1.2.0 + 7.0.2 HSA →
    `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at rocminfo; GPU not even visible.
    (librocdxg 1.2.0 is built/tested for 7.2.x; 7.0.2 HSA interface differs.
    `wsl --shutdown` did not help — not stale state.)
  - The two requirements (working WSL GPU bridge ↔ gsplat-supported ROCm)
    need incompatible ROCm versions. ~15+ install/build iterations, 6 distinct
    gsplat approaches (prebuilt wheel=gfx942 segfault, sdist=stub, upstream
    nerfstudio=no ROCm kernels, ROCm/gsplat fork build fail on 7.2.1, etc.).
  - DECISION: STOP. Consumer gfx1100 + WSL + gsplat is not viable with AMD's
    current software (officially datacenter-gfx942-only). Documented negative
    result — do NOT re-attempt without a new librocdxg/ROCm release that
    resolves the vise.
- **WHAT TO KEEP (the one durable win)**: WSL2 `ubuntu-rocm` distro on F: with
  **ROCm 7.2.1 + librocdxg + PyTorch 2.9.1+rocm6.3** = WORKING GPU compute on
  the RX 7900 XTX (verified matmul). Reusable for ANY future non-gsplat PyTorch
  GPU work. Activation: `source /root/venv/bin/activate;
  export HSA_ENABLE_DXG_DETECTION=1`. Restore script:
  `\\wsl$\ubuntu-rocm\root\restore_702_to_721.sh`. The torch HSA/HIP-symlink
  fix and librocdxg build are the non-obvious essentials (see sections above).
- **ENGINE DECISION: stay on VkSplat (on the AMD 7900 XTX).** Works, good
  results (sharp `new fotos` PSNR 21.6, bedroom 20.5), AMD-native via Vulkan,
  no version hell. gsplat quality features unreachable via AMD/ROCm here.

### gsplat via NVIDIA GTX 1070 Ti — VIABLE future path (2026-05-17)
- User HAS a GTX 1070 Ti (Pascal, **sm_61**) available to physically install.
- **gsplat SUPPORTS Pascal sm_61** — nerfstudio/gsplat default CUDA arch list
  is `90;89;86;80;75;70;61;52;37` (61 = Pascal, explicitly included). NO newer
  card needed. Install is the TRIVIAL NVIDIA path (no ROCm/WSL/version hell):
  `pip install torch --index-url .../whl/cu121` then `pip install gsplat`.
  This unlocks Mip-Splatting, appearance embeddings, exposure/pose opt — the
  features VkSplat structurally lacks.
- **REAL constraint = 1070 Ti has only 8 GB VRAM** (not the Pascal age).
  gsplat/PyTorch is far less VRAM-efficient than VkSplat (VkSplat paper: ~33%
  less VRAM than CUDA+PyTorch). 1M splats + features + Adam state on 8 GB is
  tight → may be limited to images_4 / lower cap_max. Also ~5-10× slower
  (Pascal ~8 TFLOPS vs RDNA3 ~61) → ~30-60min/run vs ~4min on VkSplat.
- **PLAN (user-chosen sequence)**: finish the lr_full VkSplat reconstruction
  first (definitive VkSplat baseline), THEN install 1070 Ti + gsplat and run
  the SAME datasets (new fotos / lr_full) head-to-head — judge by eye + LPIPS
  whether gsplat's quality features beat the 8 GB VRAM ceiling on this data.
- Current best (VkSplat) stays the reference:
  `E:\vksplat_output\sharp_mcmc1m\20260515_211727_sharp\splat.ply`.

### Why gsplat-ROCm matters (researched 2026-05-17)
- VkSplat is THE only Vulkan 3DGS *training* impl (the one we use). It has NO
  anti-aliasing/Mip-Splatting, appearance embeddings, exposure opt, depth
  supervision, pose opt, or bilateral grid — by design (speed/cross-vendor).
  Every remaining VkSplat lever is hyperparameter tuning of a feature-poor engine.
- `gsplat` (Nerfstudio backend) has ALL of those. ~3x slower than VkSplat but
  the missing features (esp. appearance embeddings + exposure opt) directly
  target the auto-exposure/WB drift in stop-and-shoot captures, and
  Mip-Splatting is the fix for the images_2 fog we currently fight with hparams.
- **GPU SUPPORT CAVEAT (official AMD gsplat docs, checked 2026-05-17)**:
  rocm.docs.amd.com/projects/gsplat requires Ubuntu 22.04/24.04 (WSL2 OK),
  ROCm 6.4.3/7.0.0. Docker is OPTIONAL (bare-metal pip path exists:
  `pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/`).
  BUT the prebuilt `amd_gsplat` wheel officially targets ONLY datacenter
  MI300X/MI325X (gfx942). RX 7900 XTX = gfx1100 (RDNA3 consumer) is NOT on the
  supported list → prebuilt wheel likely won't work. Viable path = build gsplat
  FROM SOURCE in WSL2 with `PYTORCH_ROCM_ARCH=gfx1100`. Feasible but NOT
  guaranteed (RDNA3 ROCm is rougher than datacenter; source build may fail).
  This is now the main uncertainty, NOT the OS/disk/virtualization (all resolved).

### Disk facts (verified 2026-05-17)
- F: = "OG", partition 2 of Disk 4 (WD_BLACK SN770 1TB NVMe), NTFS, 931 GB
  with ~509 GB free and ~422 GB of existing data. NOT a dedicated empty SSD.
- C: only ~56 GB free → too small for ROCm+PyTorch+gsplat+datasets.
- Plan: WSL2 distro virtual disk (ext4.vhdx) relocated onto F: (509 GB free).
  NON-DESTRUCTIVE — no formatting/partitioning; F:'s 422 GB stays intact.
  ext4 cannot be a dual-use Windows+Linux partition; the .vhdx-in-NTFS approach
  sidesteps that entirely.

### Setup Steps (BIOS step now DONE)
1. ~~Enable AMD SVM in BIOS~~ — done (verified enabled)
2. ~~Enable Virtual Machine Platform~~ — done (HypervisorPresent=True, WSL2 works)
3. Install Ubuntu 22.04 on WSL2, relocate its vhdx to F: (C: too small)
4. Run `setup_rocm_gsplat.sh` inside WSL2 (installs ROCm, ROCDXG, PyTorch, gsplat, nerfstudio)
