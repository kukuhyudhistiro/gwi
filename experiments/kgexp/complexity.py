"""B6 — operasi per piksel (analitis) untuk tahap filter + pasca-proses.

Konvensi: 1 MAC = 1 perkalian + 1 penjumlahan. Konvolusi 2D langsung k×k = k² MAC
(cv2.filter2D memakai konvolusi langsung untuk kernel kecil; untuk kernel ≥ 11×11 OpenCV
beralih ke DFT, sehingga angka analitis tidak lagi sebanding dengan wall-clock).
Gaussian di OpenCV separable: 2·m MAC untuk kernel m. Sobel 3×3 separable: 2·(3+3)
untuk gx dan gy, dihitung 12 MAC. Operasi non-MAC (sqrt, abs, compare) dicatat terpisah.
"""
from __future__ import annotations

import math

import pandas as pd


def gaussian_ksize(sigma: float) -> int:
    # aturan OpenCV untuk uint8: round(σ·3·2 + 1) | 1
    return int(round(sigma * 6 + 1)) | 1


def table(k: int, N: int, canny_sigma: float, log_sigma: float, matched_sigma: float | None = None):
    rows = [
        {"method": "GWC", "mac": 2 * N * k * k, "sqrt": N, "abs": 0, "compare": N - 1,
         "note": f"2 konvolusi {k}×{k} × {N} orientasi + |·| kompleks"},
        {"method": "GWi", "mac": N * k * k, "sqrt": 0, "abs": N, "compare": N - 1,
         "note": f"1 konvolusi {k}×{k} × {N} orientasi"},
        {"method": "GWi+ODPS", "mac": N * k * k, "sqrt": 0, "abs": N, "compare": N - 1 + 2,
         "note": "+ ODPS: 2 perbandingan + 2 baca memori/piksel"},
    ]
    for name, s in [("Canny (σ=%.2f)" % canny_sigma, canny_sigma)] + (
            [("Canny matched (σ=%.2f)" % matched_sigma, matched_sigma)] if matched_sigma else []):
        m = gaussian_ksize(s)
        rows.append({"method": name, "mac": 2 * m + 12 + 1, "sqrt": 1, "abs": 0, "compare": 2,
                     "note": f"Gaussian separable m={m} + Sobel gx,gy + |∇|² + NMS (hysteresis tidak dihitung)"})
    rows.append({"method": "Sobel", "mac": 12 + 1, "sqrt": 1, "abs": 0, "compare": 0,
                 "note": "Sobel separable gx,gy + magnitude"})
    m = 2 * math.ceil(3 * log_sigma) + 1
    rows.append({"method": f"LoG (σ={log_sigma})", "mac": 2 * m + 5, "sqrt": 0, "abs": 1, "compare": 0,
                 "note": f"Gaussian separable m={m} + Laplacian 3×3 (5 tap)"})
    df = pd.DataFrame(rows)
    ref = df.loc[df.method == "GWC", "mac"].iloc[0]
    df["mac_vs_GWC"] = df["mac"] / ref
    return df
