"""Tabel ringkas (CSV + Markdown) dan plot level-degradasi dengan whisker CI."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .stats import ods_ci, score

# Okabe–Ito (aman untuk buta warna), urutan tetap per metode
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def score_table(countsets: Dict[str, "CountSet"], cfg, extra: Dict = None) -> pd.DataFrame:
    rows = []
    for name, cs in countsets.items():
        s = score(cs.counts, cs.thresholds)
        o, (lo, hi) = ods_ci(cs.counts, cs.thresholds, cfg["stats"]["n_boot"], cfg["seed"], cfg["stats"]["ci"])
        rows.append({**(extra or {}), "method": name, "n": len(cs.image_ids), "ods": s.ods,
                     "ods_ci_lo": lo, "ods_ci_hi": hi, "ods_threshold": s.ods_threshold,
                     "ois": s.ois, "ap": s.ap})
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path, decimals=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        md = df.round(decimals).to_markdown(index=False)
    except ImportError:
        md = df.round(decimals).to_string(index=False)
    path.with_suffix(".md").write_text(md + "\n")


def plot_levels(df: pd.DataFrame, level_col: str, path: Path, xlabel: str, methods: List[str],
                title: str = "", invert_x=False, log_x=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    levels = sorted(df[level_col].unique())
    n = len(methods)
    span = 0.06 * (len(levels) - 1 if not log_x else 1)
    for i, m in enumerate(methods):
        d = df[df.method == m].sort_values(level_col)
        if d.empty:
            continue
        x = d[level_col].to_numpy(float)
        off = (i - (n - 1) / 2) * (span / max(n, 1))
        xs = x * (1 + off) if log_x else x + off * (levels[-1] - levels[0] if len(levels) > 1 else 1)
        yerr = np.vstack([d.ods - d.ods_ci_lo, d.ods_ci_hi - d.ods])
        ax.errorbar(xs, d.ods, yerr=yerr, color=PALETTE[i % 8], marker=MARKERS[i % 8], ms=5,
                    lw=1.6, capsize=2.5, label=m)
    if log_x:
        ax.set_xscale("log")
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ODS (95% bootstrap CI)")
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
