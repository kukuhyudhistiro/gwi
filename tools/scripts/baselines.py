"""
baselines.py (v2 — patched)

Classical edge detectors: Canny, Sobel, LoG, PC, ED.

PATCH NOTES vs v1:
    - ED: now requires opencv-contrib-python. Silent fallback to gradient
      magnitude removed — that fallback made ED == Canny in earlier runs.
      Now raises RuntimeError if ximgproc unavailable, allowing the user
      to either install opencv-contrib-python or exclude ED with --methods.
    - Canny: sigma increased to 1.4 (standard Canny recommendation) for
      slightly more aggressive smoothing.

Author: Kukuh Yudhistiro
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from phasepack import phasecong
    HAS_PHASEPACK = True
except ImportError:
    HAS_PHASEPACK = False

# Detect ximgproc ONCE at import time
HAS_XIMGPROC = (hasattr(cv2, "ximgproc")
                and hasattr(cv2.ximgproc, "createEdgeDrawing"))


@dataclass
class BaselineResult:
    magnitude: np.ndarray
    runtime_s: float


def run_canny_magnitude(image_norm: np.ndarray,
                        gaussian_sigma: float = 1.4) -> BaselineResult:
    """Canny-style gradient magnitude (pre-NMS, pre-hysteresis)."""
    t0 = time.perf_counter()
    img_u8 = (image_norm * 255.0).astype(np.uint8)
    smoothed = cv2.GaussianBlur(img_u8, (0, 0), gaussian_sigma)
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return BaselineResult(magnitude=mag.astype(np.float32),
                          runtime_s=time.perf_counter() - t0)


def run_sobel(image_norm: np.ndarray) -> BaselineResult:
    """Pure Sobel magnitude (no Gaussian)."""
    t0 = time.perf_counter()
    img_u8 = (image_norm * 255.0).astype(np.uint8)
    gx = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return BaselineResult(magnitude=mag.astype(np.float32),
                          runtime_s=time.perf_counter() - t0)


def run_log(image_norm: np.ndarray,
            gaussian_sigma: float = 1.5) -> BaselineResult:
    """|LoG(I)| magnitude."""
    t0 = time.perf_counter()
    img_u8 = (image_norm * 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_u8, (0, 0), gaussian_sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    mag = np.abs(log)
    return BaselineResult(magnitude=mag.astype(np.float32),
                          runtime_s=time.perf_counter() - t0)


def run_phase_congruency(image_norm: np.ndarray,
                         nscale: int = 4, norient: int = 6,
                         min_wavelength: int = 6, mult: float = 2.1,
                         sigma_onf: float = 0.55, k: float = 2.0
                         ) -> BaselineResult:
    """Phase Congruency (Kovesi 1999). Requires phasepack."""
    if not HAS_PHASEPACK:
        raise RuntimeError(
            "phasepack not installed. Install: pip install phasepack"
        )
    t0 = time.perf_counter()
    img_in = (image_norm * 255.0).astype(np.float64)
    M, m, _, _, _, _, _ = phasecong(
        img_in, nscale=nscale, norient=norient,
        minWaveLength=min_wavelength, mult=mult,
        sigmaOnf=sigma_onf, k=k,
    )
    return BaselineResult(magnitude=M.astype(np.float32),
                          runtime_s=time.perf_counter() - t0)


def run_edge_drawing(image_norm: np.ndarray) -> BaselineResult:
    """Edge Drawing via cv2.ximgproc.createEdgeDrawing.

    Raises RuntimeError if opencv-contrib-python not installed.
    """
    if not HAS_XIMGPROC:
        raise RuntimeError(
            "Edge Drawing (ED) requires opencv-contrib-python. "
            "Install: pip install opencv-contrib-python "
            "(uninstall opencv-python first if conflict occurs)"
        )
    t0 = time.perf_counter()
    img_u8 = (image_norm * 255.0).astype(np.uint8)
    ed = cv2.ximgproc.createEdgeDrawing()
    params = cv2.ximgproc.EdgeDrawing.Params()
    params.EdgeDetectionOperator = cv2.ximgproc.EDGE_DRAWING_SOBEL
    params.GradientThresholdValue = 20
    params.AnchorThresholdValue = 8
    params.MinPathLength = 10
    ed.setParams(params)
    ed.detectEdges(img_u8)
    edge_img = ed.getEdgeImage()
    mag = edge_img.astype(np.float32)
    return BaselineResult(magnitude=mag,
                          runtime_s=time.perf_counter() - t0)
