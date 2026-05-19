"""Restore (deblur) an already-staged training tier so an imperfect,
can't-reshoot capture stays usable.

Why this exists: the project's #1 quality bottleneck is motion blur from
long exposures in dim rooms (variance-of-Laplacian ~15 vs 500+ for sharp
images; see memory.md / CAPTURE_GUIDE.md). Going back to reshoot is often
impractical. This tool deblurs the *training-resolution* tier in place
so training has more recoverable detail to fit.

CRITICAL — training tier ONLY. COLMAP matches SIFT features across views;
deblur output is NOT view-consistent (each frame restored independently,
the model can synthesize different texture per view). Running this on the
COLMAP tier (images_2) would corrupt feature matching and poses
(memory.md issue #1). Point this at images_4 (or another training tier)
ONLY. Poses are already locked by the time training reads these pixels.

Primary path: GPU deep-learning deblur via a pretrained NAFNet
(GoPro-width64) or Restormer (motion_deblurring). Inference-only (no
autograd) so it sidesteps the gsplat-style backward-kernel issues on
this machine's ROCm build.

Fallback path: classical OpenCV/scipy restoration (unsharp / wiener /
denoise-sharpen). The DL path's GPU dependency is fragile here (Windows
ROCm device enumeration), so a runtime probe auto-falls-back to classical
rather than ever blocking the pipeline.

Output: a sibling <tier>_restored/ folder, JPEG Q95, EXIF-stripped, with
IDENTICAL filenames, so training consumes it by pointing the existing
--image-dir at it with no other pipeline change. The original tier is
never modified.

Usage:
  python restore_frames.py E:\\vksplat_data\\full_dataset\\images_4 --dry-run
  python restore_frames.py E:\\vksplat_data\\full_dataset\\images_4
  python restore_frames.py E:\\vksplat_data\\full_dataset\\images_4 --cpu --method unsharp
"""

import argparse
import gc
import glob
import os
import sys
import time

import cv2
import numpy as np

# Ensure the sibling helper modules resolve no matter the cwd / which
# Python env launches this (e.g. the NVIDIA CUDA env for the 1070 Ti).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the proven JPEG-strip primitive, EXIF-orient helper and the
# sharpness metric — do not reimplement (same import pattern
# select_sharp_frames.py already uses).
from prepare_dataset import save_jpeg, apply_orientation, EXIF_ORIENTATION_TAG
from select_sharp_frames import sharpness, read_color
from PIL import Image

DEFAULT_WEIGHTS = r"E:\vksplat_tools\models\NAFNet-GoPro-width64.pth"


# --------------------------------------------------------------------------
# NAFNet architecture (self-contained, inference-only).
#
# Copied to match the official megvii-research/NAFNet checkpoint state_dict
# (module/param names are checkpoint-significant). The only deviation from
# upstream: LayerNorm2d uses a plain functional forward instead of the
# custom autograd Function — identical forward math, same weight/bias
# params (so the checkpoint loads), but no custom backward (we never train,
# and a custom autograd op is an avoidable ROCm risk).
# --------------------------------------------------------------------------
def _build_nafnet(width=64, enc_blk_nums=(1, 1, 1, 28),
                  middle_blk_num=1, dec_blk_nums=(1, 1, 1, 1),
                  img_channel=3):
    import torch
    import torch.nn as nn

    class LayerNorm2d(nn.Module):
        def __init__(self, channels, eps=1e-6):
            super().__init__()
            self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
            self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
            self.eps = eps

        def forward(self, x):
            # Same math as upstream LayerNormFunction.forward, no autograd op.
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            y = (x - mu) / (var + self.eps).sqrt()
            C = x.size(1)
            return self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)

    class SimpleGate(nn.Module):
        def forward(self, x):
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2

    class NAFBlock(nn.Module):
        def __init__(self, c, DW_Expand=2, FFN_Expand=2):
            super().__init__()
            dw = c * DW_Expand
            self.conv1 = nn.Conv2d(c, dw, 1, 1, 0, groups=1, bias=True)
            self.conv2 = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw, bias=True)
            self.conv3 = nn.Conv2d(dw // 2, c, 1, 1, 0, groups=1, bias=True)
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dw // 2, dw // 2, 1, 1, 0, groups=1, bias=True),
            )
            self.sg = SimpleGate()
            ffn = FFN_Expand * c
            self.conv4 = nn.Conv2d(c, ffn, 1, 1, 0, groups=1, bias=True)
            self.conv5 = nn.Conv2d(ffn // 2, c, 1, 1, 0, groups=1, bias=True)
            self.norm1 = LayerNorm2d(c)
            self.norm2 = LayerNorm2d(c)
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()
            self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        def forward(self, inp):
            x = self.norm1(inp)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.sg(x)
            x = x * self.sca(x)
            x = self.conv3(x)
            x = self.dropout1(x)
            y = inp + x * self.beta
            x = self.conv4(self.norm2(y))
            x = self.sg(x)
            x = self.conv5(x)
            x = self.dropout2(x)
            return y + x * self.gamma

    class NAFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, groups=1, bias=True)
            self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, groups=1, bias=True)
            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            chan = width
            for num in enc_blk_nums:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
                self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
                chan *= 2
            self.middle_blks = nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)])
            for num in dec_blk_nums:
                self.ups.append(nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)))
                chan //= 2
                self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.padder_size = 2 ** len(self.encoders)

        def _pad(self, x):
            _, _, h, w = x.size()
            ph = (self.padder_size - h % self.padder_size) % self.padder_size
            pw = (self.padder_size - w % self.padder_size) % self.padder_size
            import torch.nn.functional as F
            return F.pad(x, (0, pw, 0, ph))

        def forward(self, inp):
            B, C, H, W = inp.shape
            inp = self._pad(inp)
            x = self.intro(inp)
            encs = []
            for enc, down in zip(self.encoders, self.downs):
                x = enc(x)
                encs.append(x)
                x = down(x)
            x = self.middle_blks(x)
            for dec, up, skip in zip(self.decoders, self.ups, encs[::-1]):
                x = up(x)
                x = x + skip
                x = dec(x)
            x = self.ending(x)
            x = x + inp
            return x[:, :, :H, :W]

    return NAFNet()


# --------------------------------------------------------------------------
# Device gate + readiness probe.
# --------------------------------------------------------------------------
def pick_device():
    """Return ('cuda', name) if a real GPU op succeeds, else ('cpu', reason).

    ROCm on this machine reports a torch build but FAILS at device
    enumeration ("Failed to get device count"). is_available() alone is
    not trustworthy — only an actual tensor op on the device is. We run a
    tiny conv inside try/except and treat any failure as "no GPU".
    """
    try:
        import torch
    except Exception as e:
        return "cpu", f"torch import failed: {e}"
    try:
        if not torch.cuda.is_available():
            return "cpu", "torch.cuda.is_available() == False"
        if torch.cuda.device_count() < 1:
            return "cpu", "device_count() < 1"
        import torch.nn as nn
        t = torch.zeros(1, 3, 16, 16, device="cuda")
        conv = nn.Conv2d(3, 3, 3, padding=1).to("cuda")
        with torch.no_grad():
            _ = conv(t)
        torch.cuda.synchronize()
        name = torch.cuda.get_device_name(0)
        del t, conv
        torch.cuda.empty_cache()
        return "cuda", name
    except Exception as e:
        return "cpu", f"GPU probe op failed: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# DL restore.
# --------------------------------------------------------------------------
def load_nafnet(weights_path, device):
    import torch
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"NAFNet weights not found: {weights_path}\n"
            f"Download NAFNet-GoPro-width64.pth into "
            f"{os.path.dirname(weights_path)} (see CAPTURE_GUIDE.md).")
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ck["params"] if isinstance(ck, dict) and "params" in ck else ck
    net = _build_nafnet()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Architecture mismatch would mean garbage output — fail loudly.
        raise RuntimeError(
            f"NAFNet state_dict mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. First missing: {missing[:3]}")
    net.eval().to(device)
    return net


DEFAULT_RESTORMER_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "..", "vksplat_tools", "models",
    "Restormer_pretrained", "motion_deblurring.pth")


def _build_restormer():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BiasFree_LayerNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5) * self.weight

    class WithBias_LayerNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        def forward(self, x):
            mu = x.mean(-1, keepdim=True)
            return (x - mu) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5) * self.weight + self.bias

    class LayerNorm2(nn.Module):
        def __init__(self, dim, bias=True):
            super().__init__()
            self.body = WithBias_LayerNorm(dim) if bias else BiasFree_LayerNorm(dim)
        def forward(self, x):
            h, w = x.shape[-2:]
            # b c h w -> b (h w) c -> LN -> b c h w
            x = x.flatten(2).transpose(1, 2)
            x = self.body(x)
            return x.transpose(1, 2).unflatten(2, (h, w))

    class Attention(nn.Module):
        def __init__(self, dim, num_heads, bias):
            super().__init__()
            self.num_heads = num_heads
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
            self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
            self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        def forward(self, x):
            b, c, h, w = x.shape
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            hd = c // self.num_heads
            q = q.view(b, self.num_heads, hd, h * w)
            k = k.view(b, self.num_heads, hd, h * w)
            v = v.view(b, self.num_heads, hd, h * w)
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v).view(b, c, h, w)
            return self.project_out(out)

    class FeedForward(nn.Module):
        def __init__(self, dim, ffn_exp, bias):
            super().__init__()
            hid = int(dim * ffn_exp)
            self.project_in = nn.Conv2d(dim, hid * 2, 1, bias=bias)
            self.dwconv = nn.Conv2d(hid * 2, hid * 2, 3, 1, 1, groups=hid * 2, bias=bias)
            self.project_out = nn.Conv2d(hid, dim, 1, bias=bias)
        def forward(self, x):
            x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
            return self.project_out(F.gelu(x1) * x2)

    class TransformerBlock(nn.Module):
        def __init__(self, dim, heads, ffn_exp, bias, ln_bias):
            super().__init__()
            self.norm1 = LayerNorm2(dim, ln_bias)
            self.attn = Attention(dim, heads, bias)
            self.norm2 = LayerNorm2(dim, ln_bias)
            self.ffn = FeedForward(dim, ffn_exp, bias)
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

    class Downsample(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(n, n // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))
        def forward(self, x):
            return self.body(x)

    class Upsample(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(n, n * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))
        def forward(self, x):
            return self.body(x)

    class Restormer(nn.Module):
        def __init__(self, inp_channels=3, out_channels=3, dim=48,
                     num_blocks=[4, 6, 6, 8], num_refinement=4,
                     heads=[1, 2, 4, 8], ffn_exp=2.66, bias=False,
                     ln_bias=True):
            super().__init__()
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=bias)
            self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim, heads[0], ffn_exp, bias, ln_bias) for _ in range(num_blocks[0])])
            self.down1_2 = Downsample(dim)
            self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1], ffn_exp, bias, ln_bias) for _ in range(num_blocks[1])])
            self.down2_3 = Downsample(dim * 2)
            self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2], ffn_exp, bias, ln_bias) for _ in range(num_blocks[2])])
            self.down3_4 = Downsample(dim * 4)
            self.latent = nn.Sequential(*[TransformerBlock(dim * 8, heads[3], ffn_exp, bias, ln_bias) for _ in range(num_blocks[3])])
            self.up4_3 = Upsample(dim * 8)
            self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
            self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2], ffn_exp, bias, ln_bias) for _ in range(num_blocks[2])])
            self.up3_2 = Upsample(dim * 4)
            self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
            self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1], ffn_exp, bias, ln_bias) for _ in range(num_blocks[1])])
            self.up2_1 = Upsample(dim * 2)
            self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_exp, bias, ln_bias) for _ in range(num_blocks[0])])
            self.refinement = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_exp, bias, ln_bias) for _ in range(num_refinement)])
            self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

        def forward(self, inp):
            e1 = self.patch_embed.proj(inp)
            e2 = self.encoder_level2(self.down1_2(self.encoder_level1(e1)))
            e3 = self.encoder_level3(self.down2_3(e2))
            lat = self.latent(self.down3_4(e3))
            d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(lat), e3], 1)))
            d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], 1)))
            d1 = self.decoder_level1(torch.cat([self.up2_1(d2), e1], 1))
            return self.output(self.refinement(d1)) + inp

    return Restormer()


def load_restormer(weights_path, device):
    import torch
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Restormer checkpoint not found: {weights_path}")
    net = _build_restormer()
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ck["params"] if isinstance(ck, dict) and "params" in ck else ck
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Restormer state_dict mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected")
    net.eval().to(device)
    return net


# Count of tiles whose model output was non-finite and got replaced with
# input pixels. Single-element list so _infer_tiled can mutate it; the
# caller resets it per frame and treats >0 as "this frame degraded".
_BAD_TILES = [0]


def _infer_tiled(net, rgb_u8, device, tile, overlap):
    """Run NAFNet on one HxWx3 uint8 RGB image with overlapping tiles.

    Feathered blend over the overlap removes seams. Streams tile-by-tile
    so VRAM/host RAM stay bounded (16 GB host constraint, memory.md).
    """
    import torch
    h, w = rgb_u8.shape[:2]
    inp = rgb_u8.astype(np.float32) / 255.0
    out = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w, 1), np.float32)
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - overlap), step))
    xs = list(range(0, max(1, w - overlap), step))
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            y0c, x0c = max(0, y1 - tile), max(0, x1 - tile)
            patch = inp[y0c:y1, x0c:x1]
            t = torch.from_numpy(patch).permute(2, 0, 1)[None].to(device)
            with torch.no_grad():
                o = net(t)
            o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Divergence guard. NAFNet occasionally EXPLODES on certain
            # content (observed: saturated fine-texture fabric) — the
            # tile comes back FINITE but with values like -5276..+6737
            # instead of ~[0,1]. clip() then crushes that chaos into
            # 0/1 noise => rainbow-block artifact. A healthy residual
            # restoration of a [0,1] image stays well within [-0.3,1.3];
            # if a large fraction of the tile is far outside that, the
            # model diverged here — fall back to this tile's INPUT
            # pixels (consistent, if un-deblurred) and flag the frame.
            oob = float(((o < -0.3) | (o > 1.3)).mean())
            if not np.isfinite(o).all() or oob > 0.02:
                o = patch.copy()
                _BAD_TILES[0] += 1
            np.clip(o, 0.0, 1.0, out=o)
            ph, pw = o.shape[:2]
            # Hann-ish feather so tile seams blend smoothly.
            fy = np.hanning(ph + 2)[1:-1] if ph > 1 else np.ones(ph)
            fx = np.hanning(pw + 2)[1:-1] if pw > 1 else np.ones(pw)
            mask = np.maximum(np.outer(fy, fx), 1e-3)[..., None].astype(np.float32)
            out[y0c:y1, x0c:x1] += o * mask
            wsum[y0c:y1, x0c:x1] += mask
            del t, o
    res = out / np.maximum(wsum, 1e-6)
    return np.clip(res * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Classical restore (fallback path).
# --------------------------------------------------------------------------
def _unsharp(rgb, amount=0.6, radius=1.2):
    blur = cv2.GaussianBlur(rgb, (0, 0), radius)
    out = cv2.addWeighted(rgb, 1 + amount, blur, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _wiener(rgb, psf_sigma=1.0, iters=8):
    """Mild Richardson–Lucy with a small isotropic Gaussian PSF.

    Capped iterations + clamping keep ringing bounded. Per-channel.
    """
    from scipy.signal import fftconvolve
    k = max(3, int(psf_sigma * 6) | 1)
    ax = np.arange(k) - k // 2
    g = np.exp(-(ax ** 2) / (2 * psf_sigma ** 2))
    psf = np.outer(g, g)
    psf /= psf.sum()
    psf_m = psf[::-1, ::-1]
    out = np.empty_like(rgb, np.float32)
    for c in range(3):
        obs = rgb[:, :, c].astype(np.float32) + 1e-6
        est = obs.copy()
        for _ in range(iters):
            conv = fftconvolve(est, psf, mode="same") + 1e-6
            est *= fftconvolve(obs / conv, psf_m, mode="same")
            np.clip(est, 0, 255, out=est)
        out[:, :, c] = est
    return np.clip(out, 0, 255).astype(np.uint8)


def _denoise_sharpen(rgb):
    den = cv2.fastNlMeansDenoisingColored(rgb, None, 5, 5, 7, 21)
    return _unsharp(den, amount=0.5, radius=1.2)


CLASSICAL = {"unsharp": _unsharp, "wiener": _wiener,
             "denoise-sharpen": _denoise_sharpen}


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------
def restore_one(rgb_u8, method, ctx):
    """Return restored uint8 RGB. ctx carries net/device/strength/tile.

    Resets the bad-tile counter so the caller can check `bad_tiles()`
    after this call to know if the frame degraded (a tile fell back to
    blurry input pixels => the frame is only partially deblurred).
    """
    if method in ("dl", "restormer"):
        _BAD_TILES[0] = 0
        rest = _infer_tiled(ctx["net"], rgb_u8, ctx["device"],
                            ctx["tile"], ctx["overlap"])
        s = ctx["strength"]
        if s < 1.0:
            rest = np.clip(rgb_u8 * (1 - s) + rest * s, 0, 255).astype(np.uint8)
        return rest
    return CLASSICAL[method](rgb_u8)


def bad_tiles() -> int:
    """Non-finite tiles replaced during the last restore_one() DL call."""
    return _BAD_TILES[0]


def read_rgb(path: str, exif_source: bool) -> np.ndarray:
    """Load an image as uint8 RGB.

    exif_source=False: input is an already-staged JPEG (orientation
    already baked, EXIF stripped) — fast cv2 path.
    exif_source=True: input is a RAW camera file from E:\\Downloads —
    load pixels with PIL (NO auto-rotate) then apply EXIF orientation
    manually, exactly as prepare_dataset.py does (memory.md issue #1).
    """
    if not exif_source:
        return cv2.cvtColor(read_color(path), cv2.COLOR_BGR2RGB)
    with Image.open(path) as im:
        orient = im.getexif().get(EXIF_ORIENTATION_TAG, 1)
        arr = np.array(im)
    return apply_orientation(arr, orient)


def _downscale(rgb_u8: np.ndarray, width: int) -> np.ndarray:
    """INTER_AREA resize to target width (no-op if already <= width).

    Mirrors prepare_dataset.process_one's resize so the derived tier is
    pixel-consistent with the project's normal staging.
    """
    h, w = rgb_u8.shape[:2]
    if width <= 0 or width >= w:
        return rgb_u8
    return cv2.resize(rgb_u8, (width, round(h * width / w)),
                      interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tier_dir", help="Staged TRAINING tier (e.g. .../images_4). "
                                     "NEVER point at images_2 (COLMAP tier).")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: sibling <tier>_restored)")
    p.add_argument("--method", default="dl",
                   choices=["dl", "restormer", "unsharp", "wiener", "denoise-sharpen"],
                   help="dl = GPU NAFNet (default); restormer = GPU Restormer; others = classical CPU")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="NAFNet checkpoint path")
    p.add_argument("--restormer-weights", default=DEFAULT_RESTORMER_WEIGHTS,
                   help="Restormer checkpoint path")
    p.add_argument("--dl-strength", type=float, default=0.7,
                   help="Blend DL output with original 0..1 (default 0.7); "
                        "lower = more conservative / less hallucination")
    p.add_argument("--tile", type=int, default=512, help="DL tile size px")
    p.add_argument("--overlap", type=int, default=32, help="DL tile overlap px")
    p.add_argument("--cpu", action="store_true",
                   help="Force classical CPU path (skip GPU entirely). "
                        "If --method is dl, uses unsharp.")
    p.add_argument("--dl-cpu", action="store_true",
                   help="Run the NAFNet DL model on CPU (no GPU needed). "
                        "Correct output, just slow — useful to validate "
                        "the DL path or when ROCm is unavailable.")
    p.add_argument("--force", action="store_true",
                   help="Allow restoring images_2 (COLMAP tier). Only use when "
                        "raw frames can't register at all — no clean poses to lose.")
    p.add_argument("--dry-run", action="store_true",
                   help="Probe device + report sharpness delta on a sample; "
                        "write nothing")
    p.add_argument("--sample", type=int, default=12,
                   help="Frames to process in --dry-run")
    p.add_argument("--exif-source", action="store_true",
                   help="Inputs are RAW camera files (E:\\Downloads): load "
                        "via PIL and apply EXIF orientation manually "
                        "(prepare_dataset path). Default: inputs are "
                        "already-staged oriented JPEGs.")
    p.add_argument("--full-out", default=None,
                   help="Also write the full-resolution restored image to "
                        "this dir (before any downscale). Sources untouched.")
    p.add_argument("--downscale-width", type=int, default=0,
                   help="Also emit a width-W INTER_AREA copy (the training "
                        "tier derived from the full-res restore). 0 = off.")
    p.add_argument("--downscale-out", default=None,
                   help="Dir for the --downscale-width copy "
                        "(default: the normal --out-dir / <tier>_restored).")
    args = p.parse_args()

    tier_dir = os.path.abspath(args.tier_dir)
    base = os.path.basename(tier_dir.rstrip("\\/"))
    if (base.endswith("images_2") or base == "images_2") and not args.force:
        print("REFUSING: this looks like the COLMAP tier (images_2). "
              "Restoration is training-tier ONLY — it would corrupt poses. "
              "Point at images_4 (or another training tier), or use --force "
              "if raw data can't register at all.", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(tier_dir):
        print(f"Not a directory: {tier_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted({f for pat in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
                    for f in glob.glob(os.path.join(tier_dir, pat))})
    if not files:
        print(f"No .jpg in {tier_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(files)} images in {tier_dir}")

    method = args.method
    ctx = {"strength": args.dl_strength, "tile": args.tile,
           "overlap": args.overlap}

    if method in ("dl", "restormer") and args.cpu:
        print("--cpu set: using classical 'unsharp' instead of DL.")
        method = "unsharp"
    elif method == "dl" and args.dl_cpu:
        print(f"--dl-cpu: NAFNet DL deblur on CPU (slow) "
              f"(strength={args.dl_strength}, tile={args.tile})")
        try:
            ctx["net"] = load_nafnet(args.weights, "cpu")
            ctx["device"] = "cpu"
        except Exception as e:
            print(f"  NAFNet load failed ({e}). Falling back to "
                  f"classical 'unsharp'.", file=sys.stderr)
            method = "unsharp"
    elif method in ("dl", "restormer"):
        dev, info = pick_device()
        if dev == "cuda":
            model_name = "Restormer" if method == "restormer" else "NAFNet"
            loader = (load_restormer if method == "restormer"
                      else load_nafnet)
            wpath = (args.restormer_weights if method == "restormer"
                     else args.weights)
            print(f"GPU ready: {info} -> {model_name} DL deblur "
                  f"(strength={args.dl_strength}, tile={args.tile})")
            try:
                ctx["net"] = loader(wpath, "cuda")
                ctx["device"] = "cuda"
            except Exception as e:
                print(f"  {model_name} load failed ({e}). Falling back to "
                      f"classical 'unsharp'.", file=sys.stderr)
                method = "unsharp"
        else:
            print(f"GPU NOT usable ({info}).")
            print("  Falling back to classical 'unsharp' so the pipeline "
                  "still produces a restored tier.")
            method = "unsharp"

    if args.dry_run:
        idx = np.linspace(0, len(files) - 1,
                          min(args.sample, len(files)), dtype=int)
        deltas = []
        print(f"\n--dry-run: method='{method}', {len(idx)} sample frames "
              f"(nothing written)")
        for j, i in enumerate(idx):
            rgb = read_rgb(files[i], args.exif_source)
            s0 = sharpness(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            rest = restore_one(rgb, method, ctx)
            s1 = sharpness(cv2.cvtColor(rest, cv2.COLOR_RGB2GRAY))
            deltas.append(s1 - s0)
            print(f"  [{j+1}/{len(idx)}] {os.path.basename(files[i])}: "
                  f"{s0:.0f} -> {s1:.0f} (delta {s1-s0:+.0f})")
            _cleanup()
        d = np.array(deltas)
        print(f"\nSharpness delta: median={np.median(d):+.0f} "
              f"mean={d.mean():+.0f} min={d.min():+.0f} max={d.max():+.0f}")
        improved = int((d > 0).sum())
        print(f"  improved: {improved}/{len(d)} "
              f"({'OK' if improved >= 0.7 * len(d) else 'WEAK — review method/strength'})")
        return

    out_dir = args.out_dir or (tier_dir.rstrip("\\/") + "_restored")
    os.makedirs(out_dir, exist_ok=True)
    # Optional extra outputs (full-res restored + a downscaled tier).
    full_out = args.full_out
    if full_out:
        os.makedirs(full_out, exist_ok=True)
    ds_w = args.downscale_width
    ds_out = (args.downscale_out or out_dir) if ds_w > 0 else None
    if ds_out and ds_out != out_dir:
        os.makedirs(ds_out, exist_ok=True)
    extra = []
    if full_out:
        extra.append(f"full-res -> {full_out}")
    if ds_out:
        extra.append(f"downscale w{ds_w} -> {ds_out}")
    print(f"\nMethod '{method}' -> {out_dir}  ({len(files)} frames)"
          + (("  [" + "; ".join(extra) + "]") if extra else ""))
    t0 = time.time()
    last = t0
    kept_orig = 0
    tot_bad_tiles = 0       # diverged tiles that fell back to input
    frames_w_bad = 0        # frames containing >=1 such tile
    for i, f in enumerate(files):
        stem = os.path.basename(f)
        dst = os.path.join(out_dir, stem)
        full_dst = os.path.join(full_out, stem) if full_out else None
        ds_dst = os.path.join(ds_out, stem) if ds_out else None
        # Idempotent: skip only if EVERY requested output already exists.
        wanted = [dst] + ([full_dst] if full_dst else []) \
                       + ([ds_dst] if ds_dst else [])
        if all(os.path.exists(p) for p in wanted):
            continue
        rgb = read_rgb(f, args.exif_source)
        s0 = sharpness(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        rest = restore_one(rgb, method, ctx)
        nbad = bad_tiles() if method == "dl" else 0
        s1 = sharpness(cv2.cvtColor(rest, cv2.COLOR_RGB2GRAY))
        # Two-level guard:
        #  - TILE level (inside _infer_tiled): NAFNet emits unbounded
        #    output on specific content (confirmed content-specific, not
        #    ROCm/precision/preproc — CPU, GPU fp32, GPU fp64 all behave
        #    identically). Each diverged tile already fell back to its
        #    INPUT pixels, which are clean — so `rest` is a seamless mix
        #    of deblurred-good + original-bad tiles, strictly better than
        #    all-original and artifact-free. We DO ship that (nbad>0 is
        #    normal and fine), only logging the count.
        #  - FRAME level (here): only discard the whole restored frame if
        #    the *net* result is actually worse than the source or it
        #    newly blew out highlights. Dim-room captures legitimately
        #    have bright window/lamp regions already near 255, so the
        #    clip test is the INCREASE in clipping, not absolute.
        clip0 = float((rgb >= 254).mean())
        clip1 = float((rest >= 254).mean())
        if s1 < s0 or (clip1 - clip0) > 0.03:
            rest = rgb
            kept_orig += 1
        tot_bad_tiles += nbad
        if nbad:
            frames_w_bad += 1
        # dst is the primary "<tier>_restored" output. When --full-out is
        # used, dst and full_out are typically the same full-res image;
        # the downscaled tier is derived from the SAME restored pixels.
        save_jpeg(dst, rest)
        if full_dst and full_dst != dst:
            save_jpeg(full_dst, rest)
        if ds_dst:
            save_jpeg(ds_dst, _downscale(rest, ds_w))
        _cleanup()
        now = time.time()
        if i == 0 or now - last > 3.0 or i == len(files) - 1:
            rate = (i + 1) / max(now - t0, 1e-6)
            eta = (len(files) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(files)}] {stem} {s0:.0f}->{s1:.0f} "
                  f"({rate:.2f} img/s, ETA {eta:.0f}s)")
            last = now
    print(f"\nDone in {time.time()-t0:.0f}s. "
          f"{kept_orig} frame(s) kept original (net-worse guardrail); "
          f"{tot_bad_tiles} diverged tile(s) across {frames_w_bad} "
          f"frame(s) fell back to input pixels (artifact-safe).")
    print(f"Train against it:  --image-dir {os.path.basename(out_dir)}")


def _cleanup():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
