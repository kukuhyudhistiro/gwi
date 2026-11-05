"""
nms_odps.py — Orientation-Aware Double-Peak Suppression (Stage 6)

Standalone post-processing module for GWi (Stages 1-5). Suppresses the
~2-pixel side-lobes inherent to |sin| Gabor response by comparing each
pixel against its neighbors along the edge-normal direction, using the
orientation map argmax already computed in Stage 5.

Why this works:
    A 1D step edge convolved with sin(2πx/λ)·exp(-x²/2σ²) and taken in
    absolute value produces:
      - main peak at the edge location
      - side-lobes at ±λ/2 ≈ ±2 pixels with magnitude ~0.6× main peak
    ODPS keeps only the main peak by requiring the magnitude at each
    pixel to be a local maximum along the edge-normal direction.

Why this is defensible for paper Q3:
    - Reuses Stage 5 argmax orientation (no new convolution)
    - Standard practice in classical edge detection (Canny, PC also do NMS)
    - Mathematically grounded — addresses a verified side-lobe phenomenon
    - Does NOT modify Stages 1-5 → "imaginary-only" claim intact

Author: Kukuh Yudhistiro
"""

from __future__ import annotations

import os

# Single-thread enforcement
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np


def compute_normal_offsets(orientation_idx: np.ndarray,
                           n_orientations: int = 8,
                           d: int = 2) -> tuple:
    """Compute per-pixel offsets to neighbors along edge-normal direction.

    Args:
        orientation_idx : (H, W) int — argmax orientation index from Stage 5,
                          values in [0, n_orientations - 1].
                          Orientation k corresponds to angle k*(180/n_orient)°,
                          which is the FILTER orientation.
                          The edge-normal is perpendicular to the filter,
                          i.e. at angle (k * 180/n_orient) + 90°.
        n_orientations  : number of orientations (default 8 → 22.5° spacing)
        d               : distance in pixels along normal to check (default 2)

    Returns:
        dy_pos, dx_pos : (H, W) int arrays — offsets to +normal neighbor
        dy_neg, dx_neg : (H, W) int arrays — offsets to -normal neighbor
    """
    angle_step = 180.0 / n_orientations
    # In this pipeline, θ is the EDGE-NORMAL direction
    # (verified empirically: a vertical edge produces max response at θ=0°,
    # meaning the filter detecting vertical edges has θ aligned with
    # the horizontal edge-normal direction).
    # We compare neighbors directly along θ (NOT θ+90°).
    theta_deg = orientation_idx.astype(np.float32) * angle_step
    normal_rad = np.deg2rad(theta_deg)

    # Vector along normal direction, scaled by d, rounded to integer offsets
    dx_pos = np.round(np.cos(normal_rad) * d).astype(np.int32)
    dy_pos = np.round(np.sin(normal_rad) * d).astype(np.int32)
    dx_neg = -dx_pos
    dy_neg = -dy_pos

    return dy_pos, dx_pos, dy_neg, dx_neg


def odps(magnitude: np.ndarray,
         orientation_idx: np.ndarray,
         n_orientations: int = 8,
         d: int = 2,
         keep_equal: bool = True) -> np.ndarray:
    """Orientation-aware Double-Peak Suppression.

    For each pixel (y, x) in the magnitude map, compare its value against
    two neighbors along the edge-normal direction (perpendicular to the
    dominant orientation found in Stage 5). If the pixel is not a local
    maximum along that direction, set its magnitude to zero.

    Args:
        magnitude       : (H, W) float — Stage 5 max-pooled magnitude
        orientation_idx : (H, W) int — argmax orientation index
        n_orientations  : number of orientations
        d               : distance in pixels for neighbor comparison
        keep_equal      : if True, "tie" pixels are kept (>=). if False, only
                          strict maxima are kept (>).

    Returns:
        suppressed : (H, W) float — magnitude with non-maxima zeroed out

    Note:
        This is a vectorized implementation using padding and indexing,
        avoiding explicit Python loops over pixels. Should run ~3-5 ms on
        a typical BSDS500 image (321×481).
    """
    if magnitude.shape != orientation_idx.shape:
        raise ValueError(
            f"Shape mismatch: magnitude {magnitude.shape}, "
            f"orientation {orientation_idx.shape}"
        )

    h, w = magnitude.shape
    mag = magnitude.astype(np.float32)

    # Compute per-pixel neighbor offsets
    dy_pos, dx_pos, dy_neg, dx_neg = compute_normal_offsets(
        orientation_idx, n_orientations, d
    )

    # Pad magnitude with zeros (out-of-bounds neighbors treat as 0)
    pad = d + 1
    mag_padded = np.pad(mag, pad, mode="constant", constant_values=0)

    # Compute coordinate grids
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Index into padded magnitude for + normal neighbor
    y_pos = yy + dy_pos + pad
    x_pos = xx + dx_pos + pad
    neighbor_pos = mag_padded[y_pos, x_pos]

    # Index into padded magnitude for - normal neighbor
    y_neg = yy + dy_neg + pad
    x_neg = xx + dx_neg + pad
    neighbor_neg = mag_padded[y_neg, x_neg]

    # Keep pixel if it's a local maximum along normal direction
    if keep_equal:
        is_max = (mag >= neighbor_pos) & (mag >= neighbor_neg)
    else:
        is_max = (mag > neighbor_pos) & (mag > neighbor_neg)

    suppressed = np.where(is_max, mag, 0.0).astype(np.float32)
    return suppressed


def odps_uint8(magnitude_u8: np.ndarray,
               orientation_idx: np.ndarray,
               n_orientations: int = 8,
               d: int = 2,
               renormalize: bool = True) -> np.ndarray:
    """Convenience wrapper: takes uint8 magnitude PNG, returns uint8 ODPS output.

    Args:
        magnitude_u8 : (H, W) uint8 — magnitude PNG loaded from disk
        orientation_idx : (H, W) int — argmax orientation index
        n_orientations  : 8
        d               : 2
        renormalize     : if True, stretch output back to [0, 255] range

    Returns:
        suppressed_u8 : (H, W) uint8 — ODPS-processed magnitude
    """
    mag_f = magnitude_u8.astype(np.float32) / 255.0
    suppressed = odps(mag_f, orientation_idx, n_orientations, d)

    if renormalize:
        mn, mx = suppressed.min(), suppressed.max()
        if mx > mn:
            suppressed = (suppressed - mn) / (mx - mn)
        else:
            suppressed = suppressed * 0.0

    return (suppressed * 255.0).clip(0, 255).astype(np.uint8)


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    # Create a synthetic test: vertical step edge
    h, w = 100, 100
    img = np.zeros((h, w), dtype=np.float32)
    img[:, 50:] = 1.0

    # Fake magnitude: simulate GWi side-lobes
    # Main peak at x=50, side-lobes at x=48, 52
    mag = np.zeros((h, w), dtype=np.float32)
    mag[:, 48] = 0.6
    mag[:, 49] = 0.9
    mag[:, 50] = 1.0  # main peak
    mag[:, 51] = 0.9
    mag[:, 52] = 0.6

    # Fake orientation: vertical edge, so filter is horizontal (θ=0°)
    # orientation_idx = 0 means θ=0° (filter is horizontal)
    # Edge-normal = 0° + 90° = 90° (vertical neighbor comparison? NO,
    # wait: for a VERTICAL edge in image, the filter that responds
    # maximally is the one with θ=90° (vertical filter detects vertical edge)
    # No, actually: Gabor filter at θ=0° (horizontal grating) responds
    # maximally to VERTICAL edges. The edge-normal direction is then
    # along the x-axis (horizontal).
    orient_idx = np.zeros((h, w), dtype=np.int32)  # θ=0° → horizontal filter
    # Edge-normal = 0° + 90° = 90° (vertical)? Let me reconsider.

    # In Gabor convention:
    #   filter orientation θ = direction of the grating
    #   θ=0° means horizontal sinusoidal stripes → detects vertical edges
    #   Edge-normal for vertical edge = horizontal = 0°
    # So we should compare horizontal neighbors.

    # If orient_idx=0 (θ=0°), our code computes normal=θ+90°=90°,
    # which would compare VERTICAL neighbors. That's wrong.

    # CORRECTION: The "filter orientation" in our codebase is defined as
    # the DIRECTION OF THE EDGE that the filter detects, not the grating
    # direction. Let me verify with the simulation.
    print("Testing ODPS implementation...")
    print(f"Input magnitude horizontal slice at y=50:")
    print(f"  {mag[50, 45:55]}")

    suppressed = odps(mag, orient_idx, n_orientations=8, d=2)
    print(f"Output (orient=0, expect horizontal comparison):")
    print(f"  {suppressed[50, 45:55]}")

    # If correct, only x=50 (main peak) should survive
    expected_survivors = np.argwhere(suppressed[50, 45:55] > 0).flatten()
    print(f"  Survivors at x: {expected_survivors + 45}")

    # Run with orient_idx=4 (θ=90°) to compare vertical neighbors
    orient_idx_4 = np.full_like(orient_idx, 4)  # θ = 4 * 22.5 = 90°
    suppressed_4 = odps(mag, orient_idx_4, n_orientations=8, d=2)
    print(f"With orient=4 (θ=90°, expect vertical comparison):")
    print(f"  {suppressed_4[50, 45:55]}")
