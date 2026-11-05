"""Registry metode: spesifikasi → pipeline (enhancement → preprocessing → detector).

Spesifikasi metode adalah dict yang bisa di-hash (dipakai sebagai kunci cache):
    {"name": "GWi_odps", "det": "gwi_odps", "pre": "he", "enh": None,
     "ksize": 5, "wavelength": 4.0, "odps_impl": "legacy"}

Output setiap pipeline = magnitude yang sudah dikuantisasi persis seperti
save_edge_map → PNG 8-bit → /255 di protokol naskah, sehingga angka ODS identik
dengan pipeline 02_run_unified.py + 04c_evaluate_bsds_v2.py.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from .common import odps_distance

cv2.setNumThreads(1)

import gabor_core as gc  # noqa: E402  (src/)
import nms_odps  # noqa: E402
import baselines as bl  # noqa: E402

from .odps_fast import odps_fast  # noqa: E402


# =============================================================================
# Preprocessing (replika src/preprocessing.load_and_preprocess, dari array)
# =============================================================================

def to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float64)
    g = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    return np.clip(g, 0, 255).astype(np.uint8)


def finish_pre(gray_u8: np.ndarray, pre: str) -> np.ndarray:
    """he     = HE versi PIL (pipeline naskah DSP / Tabel 5–7, dipakai M0)
    he_cv2 = HE versi OpenCV (cv2.equalizeHist), sama dengan Paper 2 (JESA 59(9), Eq. 2)
    none   = tanpa HE"""
    if pre == "he":
        gray_u8 = np.array(ImageOps.equalize(Image.fromarray(gray_u8, mode="L")))
    elif pre == "he_cv2":
        gray_u8 = cv2.equalizeHist(gray_u8)
    elif pre != "none":
        raise ValueError(pre)
    return gray_u8.astype(np.float64) / 255.0


def quantize_like_png(mag: np.ndarray) -> np.ndarray:
    """Persis save_edge_map (min-max → ×255 → uint8 truncation) lalu load /255."""
    m = mag.astype(np.float32)
    mn, mx = m.min(), m.max()
    if mx > mn:
        m = (m - mn) / (mx - mn)
    u8 = (m * 255.0).clip(0, 255).astype(np.uint8)
    return u8.astype(np.float32) / 255.0


# =============================================================================
# NMS berbasis gradien yang benar untuk koordinat citra (baris ke bawah)
# =============================================================================

def gradient_nms(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """NMS 4-arah. Catatan: src/edge_eval.non_max_suppression menukar tetangga
    diagonal (45° dibandingkan ke kanan-atas/kiri-bawah, padahal pada sumbu y ke
    bawah gradien 45° menunjuk kanan-bawah). Fungsi itu tidak dipakai di pipeline
    naskah, tetapi jangan dipakai untuk baseline baru."""
    m = mag.astype(np.float32)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    p = np.pad(m, 1)
    h, w = m.shape

    def nb(dy, dx):
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    out = np.zeros_like(m)
    bins = [((ang < 22.5) | (ang >= 157.5), (0, 1)),
            ((ang >= 22.5) & (ang < 67.5), (1, 1)),
            ((ang >= 67.5) & (ang < 112.5), (1, 0)),
            ((ang >= 112.5) & (ang < 157.5), (1, -1))]
    for mask, (dy, dx) in bins:
        keep = mask & (m >= nb(dy, dx)) & (m >= nb(-dy, -dx))
        out[keep] = m[keep]
    return out


# =============================================================================
# Detektor
# =============================================================================

def _gabor_params(spec) -> gc.GaborParams:
    return gc.GaborParams(ksize=int(spec.get("ksize", 5)),
                          wavelength=float(spec.get("wavelength", 4.0)),
                          gamma=float(spec.get("gamma", 0.5)),
                          n_orientations=int(spec.get("n_orientations", 8)))


# Border handling konvolusi Gabor. "constant" = zero padding (naskah DSP, gabor_core.py);
# "reflect" = cv2.BORDER_REFLECT (Paper 2, JESA 59(9): Sec. 3.5 dan Tabel 2, juga untuk GWC).
BORDERS = {"constant": cv2.BORDER_CONSTANT, "reflect": cv2.BORDER_REFLECT}


def gwi_argmax(img: np.ndarray, bank: gc.GWiFilterBank, border: str = "constant"):
    """Replika tahap 3–5 run_gwi_with_odps (02_run_unified.py), dengan pilihan border."""
    h, w = img.shape
    magnitude = np.zeros((h, w), dtype=np.float64)
    orient_idx = np.zeros((h, w), dtype=np.int32)
    for k, ki in enumerate(bank.kernels):
        resp = np.abs(cv2.filter2D(img, cv2.CV_64F, ki, borderType=BORDERS[border]))
        better = resp > magnitude
        magnitude[better] = resp[better]
        orient_idx[better] = k
    return magnitude, orient_idx


def gwc_magnitude(img: np.ndarray, bank: gc.GWCFilterBank, border: str) -> np.ndarray:
    """Sama dengan gabor_core.run_gwc, dengan pilihan border."""
    b = BORDERS[border]
    mag = None
    for kr, ki in zip(bank.kernels_real, bank.kernels_imag):
        rr = cv2.filter2D(img, cv2.CV_64F, kr, borderType=b)
        ri = cv2.filter2D(img, cv2.CV_64F, ki, borderType=b)
        m = np.sqrt(rr ** 2 + ri ** 2)
        mag = m if mag is None else np.maximum(mag, m)
    return mag


def canny_gradients(img: np.ndarray, sigma: float):
    """Sama dengan baselines.run_canny_magnitude (uint8 → blur → Sobel)."""
    u8 = (img * 255.0).astype(np.uint8)
    sm = cv2.GaussianBlur(u8, (0, 0), sigma) if sigma > 0 else u8
    gx = cv2.Sobel(sm, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy, np.sqrt(gx ** 2 + gy ** 2)


def matched_canny_sigma(ksize=5, wavelength=4.0, gamma=0.5) -> float:
    """σ Gaussian Canny yang menyamakan momen kedua |kernel| sepanjang normal dengan
    kernel GWi (θ = 0). Kontrol untuk confounder skala (support kernel)."""
    p = gc.GaborParams(ksize=ksize, wavelength=wavelength, gamma=gamma)
    ki = np.abs(gc.make_gabor_imaginary(p, 0.0))
    prof = ki.sum(axis=0)
    x = np.arange(len(prof)) - len(prof) // 2
    target = float(np.sqrt((prof * x ** 2).sum() / prof.sum()))

    def canny_sigma_eff(s):
        n = 61
        delta = np.zeros((n, n), np.float64)
        delta[n // 2, n // 2] = 1.0
        sm = cv2.GaussianBlur(delta, (0, 0), s) if s > 0 else delta
        kx = np.abs(cv2.Sobel(sm, cv2.CV_64F, 1, 0, ksize=3))
        pr = kx.sum(axis=0)
        xx = np.arange(n) - n // 2
        return float(np.sqrt((pr * xx ** 2).sum() / pr.sum()))

    grid = np.linspace(0.3, 4.0, 371)
    errs = [abs(canny_sigma_eff(s) - target) for s in grid]
    return float(np.round(grid[int(np.argmin(errs))], 2))


def _ed(img, grad_thr, anchor_thr):
    if not bl.HAS_XIMGPROC:
        raise RuntimeError("ED butuh opencv-contrib-python")
    u8 = (img * 255.0).astype(np.uint8)
    ed = cv2.ximgproc.createEdgeDrawing()
    prm = cv2.ximgproc.EdgeDrawing.Params()
    prm.EdgeDetectionOperator = cv2.ximgproc.EDGE_DRAWING_SOBEL
    prm.GradientThresholdValue = int(grad_thr)
    prm.AnchorThresholdValue = int(anchor_thr)
    prm.MinPathLength = 10
    ed.setParams(prm)
    ed.detectEdges(u8)
    return ed.getEdgeImage().astype(np.float32)


def build_detector(spec, cfg) -> Callable[[np.ndarray], Tuple[np.ndarray, float]]:
    """Kembalikan f(img_norm) -> (magnitude, odps_time_s)."""
    det = spec["det"]
    if det in ("gwi", "gwi_odps", "gwc"):
        prm = _gabor_params(spec)
        # Spesifikasi tanpa kunci "border" = zero padding, persis naskah DSP (kunci cache M0 tetap).
        border = spec.get("border", "constant")
        if det == "gwc":
            bank = gc.GWCFilterBank(prm)
            if border == "constant":
                return lambda img: (gc.run_gwc(img, bank).magnitude, 0.0)
            return lambda img: (gwc_magnitude(img, bank, border), 0.0)
        bank = gc.GWiFilterBank(prm)
        if det == "gwi":
            if border == "constant":
                return lambda img: (gc.run_gwi(img, bank).magnitude, 0.0)
            return lambda img: (gwi_argmax(img, bank, border)[0], 0.0)
        d = odps_distance(prm.wavelength, spec.get("odps_d", "auto"))
        impl = spec.get("odps_impl", "legacy")
        fn = nms_odps.odps if impl == "legacy" else odps_fast

        def run(img):
            mag, oi = gwi_argmax(img, bank, border)
            t = time.perf_counter()
            out = fn(mag, oi, n_orientations=prm.n_orientations, d=d)
            return out, time.perf_counter() - t
        return run
    if det == "canny_legacy":
        s = float(spec.get("sigma", 1.4))
        return lambda img: (bl.run_canny_magnitude(img, gaussian_sigma=s).magnitude, 0.0)
    if det in ("canny_nms", "canny_hyst"):
        s = spec.get("sigma", 1.4)
        if s == "matched":
            s = matched_canny_sigma(int(spec.get("ksize", 5)), float(spec.get("wavelength", 4.0)))
        s = float(s)

        def run(img):
            gx, gy, mag = canny_gradients(img, s)
            return gradient_nms(mag, gx, gy), 0.0
        return run
    if det == "sobel":
        s = float(spec.get("sigma", 0.0))
        if s == 0.0:
            return lambda img: (bl.run_sobel(img).magnitude, 0.0)
        return lambda img: (canny_gradients(img, s)[2].astype(np.float32), 0.0)
    if det == "log":
        s = float(spec.get("sigma", 1.5))
        if spec.get("legacy", False):
            return lambda img: (bl.run_log(img, gaussian_sigma=s).magnitude, 0.0)
        ks = 2 * int(np.ceil(3 * s)) + 1

        def run(img):
            u8 = (img * 255.0).astype(np.uint8)
            b = cv2.GaussianBlur(u8, (ks, ks), s)
            return np.abs(cv2.Laplacian(b, cv2.CV_64F)).astype(np.float32), 0.0
        return run
    if det == "pc":
        ns, mw = int(spec.get("nscale", 4)), int(spec.get("min_wavelength", 6))
        return lambda img: (bl.run_phase_congruency(img, nscale=ns, min_wavelength=mw).magnitude, 0.0)
    if det == "ed":
        gt_, at = spec.get("gradient_threshold", 20), spec.get("anchor_threshold", 8)
        return lambda img: (_ed(img, gt_, at), 0.0)
    raise ValueError(f"detector tidak dikenal: {det}")


# =============================================================================
# Pipeline lengkap
# =============================================================================

_ZDCE = {}


def _zerodce(cfg):
    from .zerodce import ZeroDCE
    p = cfg["paths"]["zerodce_weights"]
    if p not in _ZDCE:
        _ZDCE[p] = ZeroDCE(p)
    return _ZDCE[p]


def build_pipeline(spec, cfg) -> Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]]:
    detector = build_detector(spec, cfg)
    pre = spec.get("pre", cfg["preproc_default"])
    enh = spec.get("enh")

    def run(rgb: np.ndarray):
        tm = {}
        t0 = time.perf_counter()
        if enh == "zerodce":
            rgb = _zerodce(cfg).enhance(rgb)
        gray = to_gray_u8(rgb)
        if enh == "bm3d":
            import bm3d
            from skimage.restoration import estimate_sigma
            g = gray.astype(np.float64) / 255.0
            sig = float(estimate_sigma(g))
            gray = np.clip(np.round(255 * bm3d.bm3d(g, sigma_psd=max(sig, 1e-3))), 0, 255).astype(np.uint8)
        elif enh == "clahe":
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        t1 = time.perf_counter()
        img = finish_pre(gray, pre)
        t2 = time.perf_counter()
        mag, t_odps = detector(img)
        t3 = time.perf_counter()
        tm.update(enh_s=t1 - t0, pre_s=t2 - t1, det_s=t3 - t2 - t_odps, odps_s=t_odps,
                  total_s=t3 - t0)
        return quantize_like_png(mag), tm
    return run


# =============================================================================
# Spesifikasi standar
# =============================================================================

def gabor_from(cfg, which="GWi_odps"):
    """Parameter Gabor final.

    tuning.gabor_select = false (bawaan sejak 0.3.0): GWi tetap seperti Paper 2
    (methods_default.gabor), hasil B1 hanya analisis sensitivitas.
    tuning.gabor_select = true: pakai selected_config.json (B1) kalau ada."""
    g = dict(cfg["methods_default"]["gabor"])
    if not cfg.get("tuning", {}).get("gabor_select", True):
        return g
    sel = cfg.get("selected", {}).get(which) or cfg.get("selected", {}).get("GWi_odps")
    if sel:
        g.update(ksize=sel["ksize"], wavelength=sel["wavelength"])
    return g


def legacy_specs(cfg):
    """8 metode persis seperti naskah DSP (uji regresi M0 terhadap Tabel 5–7).

    Gabor diambil dari methods_legacy.gabor (k = 5, λ = 4, zero padding) dan HE versi PIL,
    bukan dari methods_default, supaya kunci cache M0 tidak berubah."""
    md = cfg["methods_default"]
    g = cfg.get("methods_legacy", {}).get("gabor", md["gabor"])
    return [
        {"name": "GWi", "det": "gwi", "pre": "he", **g},
        {"name": "GWi_odps", "det": "gwi_odps", "pre": "he", "odps_d": md["odps_d"], **g},
        {"name": "GWC", "det": "gwc", "pre": "he", **g},
        {"name": "Canny", "det": "canny_legacy", "pre": "he", "sigma": md["canny_sigma"]},
        {"name": "Sobel", "det": "sobel", "pre": "he", "sigma": 0.0},
        {"name": "LoG", "det": "log", "pre": "he", "sigma": md["log_sigma"], "legacy": True},
        {"name": "PC", "det": "pc", "pre": "he", **md["pc"]},
        {"name": "ED", "det": "ed", "pre": "he", **md["ed"]},
    ]


def tuned_specs(cfg, pre=None):
    """Metode utama dengan konfigurasi terpilih B1/B2 (fallback ke default)."""
    pre = pre or cfg["preproc_default"]
    sel = cfg.get("selected", {})
    md = cfg["methods_default"]
    g = gabor_from(cfg, "GWi_odps")
    gi = gabor_from(cfg, "GWi")
    gwc = sel.get("GWC", {"ksize": md["gabor"]["ksize"], "wavelength": md["gabor"]["wavelength"]})
    gwc = {"ksize": gwc["ksize"], "wavelength": gwc["wavelength"]}      # buang ods_trainval
    border = {"border": md["gabor"]["border"]} if "border" in md["gabor"] else {}
    canny = sel.get("Canny", {"sigma": md["canny_sigma"]})
    return [
        {"name": "GWi_odps", "det": "gwi_odps", "pre": pre, "odps_d": md["odps_d"], **g},
        {"name": "GWi", "det": "gwi", "pre": pre, **gi},
        {"name": "GWC_tuned", "det": "gwc", "pre": pre, "gamma": 0.5, "n_orientations": 8, **gwc, **border},
        {"name": "Canny_tuned", "det": "canny_hyst", "pre": pre, "sigma": canny["sigma"],
         "hyst_low_ratio": cfg["tuning"]["canny_low_ratio"]},
        {"name": "Canny_matched", "det": "canny_nms", "pre": pre, "sigma": "matched",
         "ksize": g["ksize"], "wavelength": g["wavelength"]},
    ]
