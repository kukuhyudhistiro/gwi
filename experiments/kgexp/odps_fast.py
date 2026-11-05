"""B8 — ODPS tervektorisasi berbasis slicing per orientasi.

Versi lama (src/nms_odps.odps) membangun meshgrid, menghitung cos/sin per piksel, dan
fancy-indexing dua kali setiap citra. Di sini offset (dx, dy) dihitung SEKALI per
orientasi (rumus pembulatan sama persis), lalu untuk tiap orientasi k dibuat dua view
bergeser dari array ber-padding (tanpa salinan) dan dibandingkan hanya pada piksel
dengan argmax = k. Hasilnya harus identik bit-per-bit dengan versi lama
(diuji di tests/test_odps_fast.py dan scripts/b8_odps.py).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def orientation_offsets(n_orientations: int, d: int):
    k = np.arange(n_orientations, dtype=np.float32)
    theta = np.deg2rad(k * np.float32(180.0 / n_orientations))
    dx = np.round(np.cos(theta) * d).astype(np.int32)
    dy = np.round(np.sin(theta) * d).astype(np.int32)
    return tuple(zip(dy.tolist(), dx.tolist()))


def odps_fast(magnitude: np.ndarray, orientation_idx: np.ndarray,
              n_orientations: int = 8, d: int = 2, keep_equal: bool = True) -> np.ndarray:
    mag = magnitude.astype(np.float32, copy=False)
    h, w = mag.shape
    pad = d + 1
    mp = np.pad(mag, pad, mode="constant")
    keep = np.zeros((h, w), dtype=bool)
    for k, (dy, dx) in enumerate(orientation_offsets(n_orientations, d)):
        sel = orientation_idx == k
        if not sel.any():
            continue
        pos = mp[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
        neg = mp[pad - dy:pad - dy + h, pad - dx:pad - dx + w]
        if keep_equal:
            ok = (mag >= pos) & (mag >= neg)
        else:
            ok = (mag > pos) & (mag > neg)
        keep |= sel & ok
    return np.where(keep, mag, np.float32(0.0)).astype(np.float32)
