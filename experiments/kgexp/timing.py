"""Harness runtime yang sama untuk semua metode (B4, B7b, B8).

Aturan: 1 thread (OpenCV/BLAS), warm-up `warmup` citra tanpa dicatat, lalu setiap
citra dijalankan `repeats` kali dan diambil median per tahap. Laporan = median dan
mean ± SD lintas citra. Evaluasi akurasi tidak pernah dijalankan dalam harness ini.
"""
from __future__ import annotations

import platform
from typing import List

import numpy as np
import pandas as pd

from . import degrade
from .data import load_rgb
from .methods import build_pipeline


def run_timing(cfg, samples, specs: List[dict], condition="clean") -> pd.DataFrame:
    import cv2
    cv2.setNumThreads(1)
    t = cfg["timing"]
    cond = degrade.Condition.parse(condition)
    pipes = [build_pipeline(s, cfg) for s in specs]
    imgs = [degrade.apply(load_rgb(s), cond, cfg, s.image_id) for s in samples]
    for rgb in imgs[: t["warmup"]]:
        for p in pipes:
            p(rgb)
    rows = []
    for s, rgb in zip(samples, imgs):
        for spec, p in zip(specs, pipes):
            reps = [p(rgb)[1] for _ in range(t["repeats"])]
            med = {k: float(np.median([r[k] for r in reps])) for k in reps[0]}
            rows.append({"image_id": s.image_id, "method": spec["name"],
                         "h": rgb.shape[0], "w": rgb.shape[1], **{k + "_ms": v * 1e3 for k, v in med.items()}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.endswith("_ms")]
    g = df.groupby("method")[cols]
    out = g.median().add_suffix("_median").join(g.mean().add_suffix("_mean")).join(g.std().add_suffix("_sd"))
    out["n"] = df.groupby("method").size()
    out["cpu"] = platform.processor() or platform.machine()
    return out.reset_index()
