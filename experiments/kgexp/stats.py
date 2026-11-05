"""B3 — skor dari cache count, bootstrap CI, uji berpasangan, Holm.

Notasi (per citra i, threshold t):
    P_t = Σ_i TPpred_it / Σ_i Npred_it ,  R_t = Σ_i TPgt_it / Σ_i Ngt_it
    F_t = 2 P_t R_t / (P_t + R_t) ,       ODS = max_t F_t
Bootstrap: bobot multinomial w ~ Mult(n, 1/n) per replikasi; count teragregasi = w·C.
ΔODS berpasangan memakai w yang sama untuk kedua metode (resample indeks citra sama).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as sps

EPS = 1e-10


def _prf(c):  # c[..., 4] = (TPgt, Ngt, TPpred, Npred)
    p = c[..., 2] / np.maximum(c[..., 3], EPS)
    r = c[..., 0] / np.maximum(c[..., 1], EPS)
    f = 2 * p * r / np.maximum(p + r, EPS)
    return p, r, f


@dataclass
class Score:
    ods: float
    ods_t_idx: int
    ods_threshold: float
    ods_p: float
    ods_r: float
    ois: float
    ap: float


def score(counts: np.ndarray, thresholds: np.ndarray) -> Score:
    """Identik dengan aggregate_scores di 04c."""
    tot = counts.sum(axis=0).astype(np.float64)
    p, r, f = _prf(tot)
    i = int(np.argmax(f))
    pi, ri, fi = _prf(counts.astype(np.float64))
    ois = float(np.mean(fi.max(axis=1)))
    aps = []
    for k in range(counts.shape[0]):
        o = np.argsort(ri[k])
        aps.append(float(np.trapezoid(pi[k][o], ri[k][o])))
    return Score(float(f[i]), i, float(thresholds[i]), float(p[i]), float(r[i]), ois,
                 float(np.mean(aps)))


def f_at(counts: np.ndarray, t_idx: int) -> float:
    """F dataset pada threshold tetap (dipakai untuk 'threshold transfer')."""
    return float(_prf(counts[:, t_idx].sum(axis=0).astype(np.float64))[2])


def per_image_f(counts: np.ndarray, t_idx: int) -> np.ndarray:
    return _prf(counts[:, t_idx].astype(np.float64))[2]


def boot_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)


def boot_ods(counts: np.ndarray, W: np.ndarray) -> np.ndarray:
    n, T, _ = counts.shape
    agg = (W @ counts.reshape(n, T * 4).astype(np.float64)).reshape(-1, T, 4)
    return _prf(agg)[2].max(axis=1)


def ci(x: np.ndarray, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def ods_ci(counts, thresholds, n_boot=1000, seed=42, level=0.95):
    W = boot_weights(counts.shape[0], n_boot, seed)
    return score(counts, thresholds).ods, ci(boot_ods(counts, W), level)


def paired_delta(ca: np.ndarray, cb: np.ndarray, thresholds, n_boot=1000, seed=42, level=0.95):
    """ΔODS = ODS(a) − ODS(b) dengan CI bootstrap berpasangan."""
    if ca.shape != cb.shape:
        raise ValueError("count set tidak sejajar (jumlah citra/threshold berbeda)")
    W = boot_weights(ca.shape[0], n_boot, seed)
    d = boot_ods(ca, W) - boot_ods(cb, W)
    point = score(ca, thresholds).ods - score(cb, thresholds).ods
    lo, hi = ci(d, level)
    return {"delta": point, "ci_lo": lo, "ci_hi": hi,
            "p_boot": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def wilcoxon_rb(fa: np.ndarray, fb: np.ndarray):
    """Wilcoxon signed-rank dua sisi + rank-biserial r (matched pairs)."""
    d = fa - fb
    nz = d[d != 0]
    if len(nz) == 0:
        return {"W": 0.0, "p": 1.0, "r_rb": 0.0, "n": int(len(d))}
    res = sps.wilcoxon(fa, fb, zero_method="wilcox", alternative="two-sided")
    ranks = sps.rankdata(np.abs(nz))
    wp, wn = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return {"W": float(res.statistic), "p": float(res.pvalue),
            "r_rb": float((wp - wn) / (wp + wn)), "n": int(len(d))}


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def compare(csa, csb, cfg) -> Dict:
    """Satu baris perbandingan lengkap (ΔODS + CI berpasangan + Wilcoxon per citra)."""
    if not np.array_equal(csa.image_ids, csb.image_ids):
        raise ValueError(f"{csa.name} vs {csb.name}: image_ids berbeda")
    st = cfg["stats"]
    sa, sb = score(csa.counts, csa.thresholds), score(csb.counts, csb.thresholds)
    pd_ = paired_delta(csa.counts, csb.counts, csa.thresholds, st["n_boot"], cfg["seed"], st["ci"])
    wx = wilcoxon_rb(per_image_f(csa.counts, sa.ods_t_idx), per_image_f(csb.counts, sb.ods_t_idx))
    return {"a": csa.name, "b": csb.name, "ods_a": sa.ods, "ods_b": sb.ods, **pd_,
            "wilcoxon_p": wx["p"], "r_rb": wx["r_rb"], "n": wx["n"]}
