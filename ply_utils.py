"""
PLY read/write utilities for 3D Gaussian Splatting models.
Handles the binary PLY format used by VkSplat (standard 3DGS convention).
"""

import struct
import numpy as np


# SH permutation: maps internal interleaved order (coeff, channel) to PLY grouped-by-channel order.
# Internal: [R_dc, G_dc, B_dc, R_1, G_1, B_1, R_2, G_2, B_2, ...]
# PLY: [f_dc_0(R), f_dc_1(G), f_dc_2(B), f_rest_0..14(R), f_rest_15..29(G), f_rest_30..44(B)]
SH_PERMUTE = [i if i < 3 else (i % 3) * 15 + (i // 3) + 2 for i in range(48)]
SH_PERMUTE_INV = [0] * 48
for _i, _p in enumerate(SH_PERMUTE):
    SH_PERMUTE_INV[_p] = _i


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inverse_sigmoid(x):
    return np.log(x / (1.0 - x))


def read_ply(path):
    """
    Read a 3DGS PLY file (as written by VkSplat).
    
    Returns dict with:
        xyz: (N, 3) float32 - positions
        sh_coeffs: (N, 16, 3) float32 - SH coefficients in internal interleaved order
        opacity: (N,) float32 - linear opacity [0, 1]
        scales: (N, 3) float32 - linear scales
        rotations: (N, 4) float32 - quaternions
    """
    with open(path, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        num_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property float'):
                properties.append(line.split()[-1])

        assert num_vertices > 0, f"No vertices found in PLY header"
        num_props = len(properties)
        assert num_props == 59, f"Expected 59 float properties, got {num_props}"

        # Read binary data
        data = np.frombuffer(f.read(num_vertices * num_props * 4), dtype=np.float32)
        data = data.reshape(num_vertices, num_props)

    # Parse fields based on standard 3DGS PLY layout:
    # x, y, z (3)
    # f_dc_0..2, f_rest_0..44 (48)
    # opacity (1)
    # scale_0..2 (3)
    # rot_0..3 (4)
    xyz = data[:, 0:3].copy()
    sh_ply = data[:, 3:51].copy()  # 48 SH in PLY order
    opacity_logit = data[:, 51].copy()
    scales_log = data[:, 52:55].copy()
    rotations = data[:, 55:59].copy()

    # Convert SH from PLY order to internal interleaved order
    # PLY[permute[i]] = internal[i], so internal[i] = PLY[permute[i]]
    sh_internal = sh_ply[:, SH_PERMUTE]
    sh_coeffs = sh_internal.reshape(num_vertices, 16, 3)

    # Convert scales from log to linear
    scales = np.exp(scales_log)

    # Convert opacity from logit to linear
    opacity = sigmoid(opacity_logit)

    return {
        'xyz': xyz,
        'sh_coeffs': sh_coeffs,
        'opacity': opacity,
        'scales': scales,
        'rotations': rotations,
        'num_splats': num_vertices,
    }


def write_ply(path, xyz, sh_coeffs, opacity, scales, rotations):
    """
    Write a 3DGS PLY file in VkSplat-compatible format.
    
    Args:
        xyz: (N, 3) float32 - positions
        sh_coeffs: (N, 16, 3) float32 - SH in internal interleaved order
        opacity: (N,) float32 - linear opacity [0, 1]
        scales: (N, 3) float32 - linear scales
        rotations: (N, 4) float32 - quaternions
    """
    num_splats = xyz.shape[0]
    assert sh_coeffs.shape == (num_splats, 16, 3)
    assert opacity.shape == (num_splats,)
    assert scales.shape == (num_splats, 3)
    assert rotations.shape == (num_splats, 4)

    # Convert SH from internal interleaved order to PLY order
    sh_internal_flat = sh_coeffs.reshape(num_splats, 48)
    sh_ply = np.zeros_like(sh_internal_flat)
    for i in range(48):
        sh_ply[:, SH_PERMUTE[i]] = sh_internal_flat[:, i]

    # Convert scales to log
    scales_log = np.log(np.clip(scales, 1e-8, None))

    # Convert opacity to logit
    opacity_clamped = np.clip(opacity, 1e-6, 1.0 - 1e-6)
    opacity_logit = inverse_sigmoid(opacity_clamped)

    # Build property names
    prop_names = ['x', 'y', 'z']
    prop_names += ['f_dc_0', 'f_dc_1', 'f_dc_2']
    prop_names += [f'f_rest_{i}' for i in range(45)]
    prop_names += ['opacity', 'scale_0', 'scale_1', 'scale_2']
    prop_names += ['rot_0', 'rot_1', 'rot_2', 'rot_3']

    # Write
    with open(path, 'wb') as f:
        header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += f"element vertex {num_splats}\n"
        for name in prop_names:
            header += f"property float {name}\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))

        # Pack data: xyz(3) + sh(48) + opacity(1) + scales(3) + rotations(4) = 59
        out = np.zeros((num_splats, 59), dtype=np.float32)
        out[:, 0:3] = xyz
        out[:, 3:51] = sh_ply
        out[:, 51] = opacity_logit
        out[:, 52:55] = scales_log
        out[:, 55:59] = rotations
        f.write(out.tobytes())


def print_ply_stats(data):
    """Print statistics about a loaded PLY for debugging."""
    n = data['num_splats']
    nan_xyz = np.isnan(data['xyz']).any(axis=1).sum()
    print(f"  Num splats: {n}")
    if nan_xyz > 0:
        print(f"  WARNING: {nan_xyz} splats have NaN positions")
    valid = ~np.isnan(data['xyz']).any(axis=1)
    if valid.any():
        print(f"  XYZ range: [{data['xyz'][valid].min(axis=0)}] to [{data['xyz'][valid].max(axis=0)}]")
    print(f"  Opacity: min={np.nanmin(data['opacity']):.4f}, max={np.nanmax(data['opacity']):.4f}, "
          f"mean={np.nanmean(data['opacity']):.4f}, median={np.nanmedian(data['opacity']):.4f}")
    print(f"  Scales: min={np.nanmin(data['scales']):.6f}, max={np.nanmax(data['scales']):.4f}, "
          f"mean={np.nanmean(data['scales']):.6f}")
    print(f"  SH DC range: [{np.nanmin(data['sh_coeffs'][:, 0, :]):.3f}, {np.nanmax(data['sh_coeffs'][:, 0, :]):.3f}]")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ply_utils.py <path_to_ply> [--roundtrip output.ply]")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Reading: {path}")
    data = read_ply(path)
    print_ply_stats(data)

    if '--roundtrip' in sys.argv:
        out_path = sys.argv[sys.argv.index('--roundtrip') + 1]
        print(f"\nWriting roundtrip: {out_path}")
        write_ply(out_path, data['xyz'], data['sh_coeffs'], data['opacity'], data['scales'], data['rotations'])
        print("Verifying roundtrip...")
        data2 = read_ply(out_path)
        assert np.allclose(data['xyz'], data2['xyz'], atol=1e-6, equal_nan=True)
        assert np.allclose(data['sh_coeffs'], data2['sh_coeffs'], atol=1e-5, equal_nan=True)
        assert np.allclose(data['opacity'], data2['opacity'], atol=1e-4, equal_nan=True)
        assert np.allclose(data['scales'], data2['scales'], rtol=1e-4, equal_nan=True)
        assert np.allclose(data['rotations'], data2['rotations'], atol=1e-6, equal_nan=True)
        print("Roundtrip OK!")
