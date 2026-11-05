"""
gabor_core.py

Core Gabor wavelet implementation: kernel generators and inference pipelines.

This module is the single source of truth for:
- GWC (Gabor Wavelet Complex)  — real + imaginary components
- GWi (Gabor Wavelet Imaginary) — imaginary component only (proposed)

All convolutions use cv2.filter2D with BORDER_CONSTANT for parity with the
original Paper 1 experiments. The functions return raw magnitude responses
WITHOUT thresholding — thresholding is handled separately by edge_eval.py
to support per-image threshold sweep required by BSDS500 protocol.

Author: Kukuh Yudhistiro
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


# ============================================================================
# Global parameter defaults (kept in one place for traceability)
# ============================================================================

@dataclass(frozen=True)
class GaborParams:
    """Immutable parameter container for Gabor wavelet pipeline.
    
    Defaults selected via cross-dataset ablation (BSDS500, BIPED, UDED):
    k=5, λ=4, n=8 achieves highest mean ODS (0.5304) at lowest runtime
    among top-performing configurations.
    """

    ksize: int = 5
    wavelength: float = 4.0
    gamma: float = 0.5
    psi_deg: float = 0.0
    n_orientations: int = 8

    @property
    def sigma(self) -> float:
        """sigma = 0.56 * wavelength (Kruizinga & Petkov 1999 rule)."""
        return 0.56 * self.wavelength

    @property
    def orientations_deg(self) -> List[float]:
        """8 orientations at 22.5 degree spacing: 0, 22.5, ..., 157.5."""
        return [i * (180.0 / self.n_orientations)
                for i in range(self.n_orientations)]


# ============================================================================
# Kernel generators
# ============================================================================

def _gabor_grids(ksize: int, theta_rad: float):
    """Pre-compute rotated coordinate grids for a kernel.

    Returns:
        x_t, y_t : rotated coordinate grids (size ksize x ksize)
    """
    half = ksize // 2
    x, y = np.meshgrid(np.arange(-half, half + 1),
                       np.arange(-half, half + 1))
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    x_t = x * cos_t + y * sin_t
    y_t = -x * sin_t + y * cos_t
    return x_t, y_t


def make_gabor_complex_pair(params: GaborParams,
                            theta_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (real, imaginary) Gabor kernel pair for one orientation.

    Both kernels share the same Gaussian envelope. Real = cos(...), Imag = sin(...).

    Returns:
        kr : real component kernel  (ksize x ksize, float64)
        ki : imag component kernel  (ksize x ksize, float64)
    """
    ksize = params.ksize if params.ksize % 2 == 1 else params.ksize + 1
    theta = np.deg2rad(theta_deg)
    psi = np.deg2rad(params.psi_deg)

    x_t, y_t = _gabor_grids(ksize, theta)
    gauss = np.exp(-0.5 * (x_t ** 2 + (params.gamma ** 2) * y_t ** 2)
                   / (params.sigma ** 2))
    sinusoid_arg = 2.0 * np.pi * x_t / params.wavelength + psi

    kr = (gauss * np.cos(sinusoid_arg)).astype(np.float64)
    ki = (gauss * np.sin(sinusoid_arg)).astype(np.float64)
    return kr, ki


def make_gabor_imaginary(params: GaborParams,
                         theta_deg: float) -> np.ndarray:
    """Generate imaginary-only Gabor kernel for one orientation."""
    ksize = params.ksize if params.ksize % 2 == 1 else params.ksize + 1
    theta = np.deg2rad(theta_deg)
    psi = np.deg2rad(params.psi_deg)

    x_t, y_t = _gabor_grids(ksize, theta)
    gauss = np.exp(-0.5 * (x_t ** 2 + (params.gamma ** 2) * y_t ** 2)
                   / (params.sigma ** 2))
    return (gauss * np.sin(2.0 * np.pi * x_t / params.wavelength + psi)
            ).astype(np.float64)


# ============================================================================
# Filter banks (pre-computed once, reused per image)
# ============================================================================

class GWCFilterBank:
    """Pre-computed GWC filter bank for fast per-image inference.

    Building the kernels is cheap (8 * 2 = 16 kernels of size 7x7), but doing
    it inside an inner loop over 280 images wastes time and obscures runtime
    measurements. Build once, apply many times.
    """

    def __init__(self, params: GaborParams = GaborParams()):
        self.params = params
        self.kernels_real: List[np.ndarray] = []
        self.kernels_imag: List[np.ndarray] = []
        for theta in params.orientations_deg:
            kr, ki = make_gabor_complex_pair(params, theta)
            self.kernels_real.append(kr)
            self.kernels_imag.append(ki)


class GWiFilterBank:
    """Pre-computed GWi filter bank (imaginary-only)."""

    def __init__(self, params: GaborParams = GaborParams()):
        self.params = params
        self.kernels: List[np.ndarray] = []
        for theta in params.orientations_deg:
            self.kernels.append(make_gabor_imaginary(params, theta))


# ============================================================================
# Pipelines (raw magnitude, NO thresholding)
# ============================================================================

@dataclass
class GaborResult:
    """Output of a single Gabor inference run."""

    magnitude: np.ndarray   # combined max-over-orientations magnitude
    runtime_s: float        # wall-clock seconds (convolution + max only)
    per_orient: List[np.ndarray]  # per-orientation magnitudes (for analysis)


def run_gwc(image_norm: np.ndarray, bank: GWCFilterBank) -> GaborResult:
    """Run GWC pipeline on a normalized [0,1] grayscale image.

    For each orientation:
        m_theta = sqrt( (I * kr)^2 + (I * ki)^2 )
    Combined response: M(x,y) = max_theta m_theta(x,y).

    Returns raw magnitude WITHOUT thresholding.
    """
    t0 = time.perf_counter()
    mags: List[np.ndarray] = []
    for kr, ki in zip(bank.kernels_real, bank.kernels_imag):
        resp_r = cv2.filter2D(image_norm, cv2.CV_64F, kr,
                              borderType=cv2.BORDER_CONSTANT)
        resp_i = cv2.filter2D(image_norm, cv2.CV_64F, ki,
                              borderType=cv2.BORDER_CONSTANT)
        mags.append(np.sqrt(resp_r ** 2 + resp_i ** 2))
    combined = np.max(np.stack(mags, axis=0), axis=0)
    runtime = time.perf_counter() - t0
    return GaborResult(magnitude=combined, runtime_s=runtime, per_orient=mags)


def run_gwi(image_norm: np.ndarray, bank: GWiFilterBank) -> GaborResult:
    """Run GWi pipeline on a normalized [0,1] grayscale image.

    For each orientation:
        m_theta = |I * ki|
    Combined response: M(x,y) = max_theta m_theta(x,y).

    Returns raw magnitude WITHOUT thresholding.
    """
    t0 = time.perf_counter()
    mags: List[np.ndarray] = []
    for ki in bank.kernels:
        resp = cv2.filter2D(image_norm, cv2.CV_64F, ki,
                            borderType=cv2.BORDER_CONSTANT)
        mags.append(np.abs(resp))
    combined = np.max(np.stack(mags, axis=0), axis=0)
    runtime = time.perf_counter() - t0
    return GaborResult(magnitude=combined, runtime_s=runtime, per_orient=mags)


# ============================================================================
# Normalization helper (output to [0,1] for downstream NMS/thresholding)
# ============================================================================

def normalize_magnitude(mag: np.ndarray) -> np.ndarray:
    """Min-max normalize magnitude map to [0, 1] for thresholding/NMS.

    Returns float32 to save memory in batch processing.
    """
    mn, mx = mag.min(), mag.max()
    if mx <= mn:
        return np.zeros_like(mag, dtype=np.float32)
    return ((mag - mn) / (mx - mn)).astype(np.float32)
