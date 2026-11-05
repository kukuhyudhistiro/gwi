"""Evaluator cepat yang menghasilkan count IDENTIK dengan 04c_evaluate_bsds_v2.py.

Mengapa identik (bukan sekadar "mirip"):
  1. Titik prediksi = np.argwhere(thin(pred >= t)), sama seperti 04c.
  2. Kandidat per titik = KDTree.query_ball_point pada titik yang sama. Di sini query dijalankan
     SEKALI per citra untuk semua piksel yang mungkin muncul di threshold mana pun (superset),
     lalu baris yang dibutuhkan diambil per threshold. Hasil query per titik tidak bergantung
     pada titik lain dalam batch, dan query multi-titik mengembalikan indeks terurut seperti 04c.
  3. Urutan pasangan (pred i, lalu kandidat GT sesuai urutan query) dan biaya
     sqrt(dx² + dy²) sama persis dengan loop np.linalg.norm di 04c (operasi pada bilangan bulat
     kecil sehingga eksak), jadi np.argsort menghasilkan urutan yang sama.
  4. Greedy berurutan diganti "locally dominant edges": pada urutan total yang sama, sisi yang
     berperingkat terkecil di kedua ujungnya pasti diterima greedy; terima semua sisi itu,
     buang sisi yang bersinggungan, ulangi. Hasilnya sama dengan greedy berurutan.
  Kesetaraan diuji per citra per threshold terhadap 04c di tests/test_fast_eval.py dan
  scripts/m0_verify_fast_eval.py (data asli).
Kasus tepi: kalau hanya ada 1 titik prediksi, 04c memakai query satu titik (tidak terurut);
count tetap sama karena satu titik paling banyak cocok dengan satu GT.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial import KDTree
from skimage.morphology import thin as sk_thin


class _AnnotatorIndex:
    """Kandidat GT per piksel (format CSR atas indeks linear piksel) untuk satu anotator."""

    def __init__(self, gt_bool: np.ndarray, union_pts: np.ndarray, width: int, r: float):
        self.gt_pts = np.argwhere(sk_thin(gt_bool))
        self.n_gt = len(self.gt_pts)
        self.width = width
        self.pix_lin = union_pts[:, 0] * width + union_pts[:, 1]     # terurut (row-major)
        if self.n_gt == 0 or len(union_pts) == 0:
            self.indptr = np.zeros(len(union_pts) + 1, np.int64)
            self.indices = np.zeros(0, np.int64)
            return
        tree = KDTree(self.gt_pts)
        cand = tree.query_ball_point(union_pts, r=r, return_sorted=True)
        lens = np.fromiter((len(c) for c in cand), np.int64, count=len(cand))
        self.indptr = np.concatenate([[0], np.cumsum(lens)])
        self.indices = (np.fromiter((g for c in cand for g in c), np.int64, count=int(lens.sum()))
                        if lens.sum() else np.zeros(0, np.int64))

    def pairs(self, pred_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lin = pred_pts[:, 0] * self.width + pred_pts[:, 1]
        pos = np.searchsorted(self.pix_lin, lin)
        starts, ends = self.indptr[pos], self.indptr[pos + 1]
        lens = ends - starts
        tot = int(lens.sum())
        if tot == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        rows = np.repeat(np.arange(len(pred_pts)), lens)
        offs = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(tot)
        return rows, self.indices[offs]


def _greedy_by_order(rows, cols, cost, n_p, n_g):
    """Greedy matching menurut np.argsort(cost) — setara dengan loop di 04c."""
    order = np.argsort(cost)                 # default kind, sama dengan 04c
    r, c = rows[order], cols[order]
    rank = np.arange(len(order))
    pm = np.zeros(n_p, bool)
    gm = np.zeros(n_g, bool)
    big = len(order)
    while len(rank):
        bp = np.full(n_p, big, np.int64)
        np.minimum.at(bp, r, rank)
        bg = np.full(n_g, big, np.int64)
        np.minimum.at(bg, c, rank)
        dom = (bp[r] == rank) & (bg[c] == rank)
        pm[r[dom]] = True
        gm[c[dom]] = True
        keep = ~pm[r] & ~gm[c]
        r, c, rank = r[keep], c[keep], rank[keep]
    return pm, int(gm.sum())


class FastImageEvaluator:
    def __init__(self, gts: List[np.ndarray], union_mask: np.ndarray, max_dist_frac: float):
        h, w = union_mask.shape
        self.r = max_dist_frac * np.sqrt(h * h + w * w)
        union_pts = np.argwhere(union_mask)
        self.ann = [_AnnotatorIndex(g, union_pts, w, self.r) for g in gts]
        self.total_gt = int(sum(a.n_gt for a in self.ann))

    def counts(self, pred_bool: np.ndarray) -> Tuple[int, int, int, int]:
        """(count_r, sum_r, count_p, sum_p) — urutan sama dengan evaluate_threshold_bsds."""
        pred_pts = np.argwhere(sk_thin(pred_bool))
        n_pred = len(pred_pts)
        if n_pred == 0:
            return 0, self.total_gt, 0, 0
        any_m = np.zeros(n_pred, bool)
        tp_gt = 0
        for a in self.ann:
            if a.n_gt == 0:
                continue
            rows, cols = a.pairs(pred_pts)
            if len(rows) == 0:
                continue
            d = (pred_pts[rows] - a.gt_pts[cols]).astype(np.float64)
            cost = np.sqrt((d * d).sum(axis=1))
            pm, ng = _greedy_by_order(rows, cols, cost, n_pred, a.n_gt)
            any_m |= pm
            tp_gt += ng
        return tp_gt, self.total_gt, int(any_m.sum()), n_pred


def evaluate_image_thresholds_fast(pred: np.ndarray, gts: List[np.ndarray], thresholds: np.ndarray,
                                   max_dist_frac: float) -> np.ndarray:
    """Pengganti evaluate_image_thresholds (04c) dengan hasil identik."""
    fe = FastImageEvaluator(gts, pred >= float(np.min(thresholds)), max_dist_frac)
    return np.array([fe.counts(pred >= t) for t in thresholds], dtype=np.int64)


def evaluate_binary_maps_fast(maps: Iterable[np.ndarray], gts: List[np.ndarray],
                              max_dist_frac: float) -> np.ndarray:
    """Untuk peta biner arbitrer per threshold (mis. hysteresis Canny)."""
    maps = list(maps)
    union = np.logical_or.reduce(maps) if maps else None
    fe = FastImageEvaluator(gts, union, max_dist_frac)
    return np.array([fe.counts(m) for m in maps], dtype=np.int64)
