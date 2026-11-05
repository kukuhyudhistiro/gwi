"""Evaluator cepat harus identik dengan 04c (count per threshold, selisih 0)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
from skimage.filters import apply_hysteresis_threshold  # noqa: E402

from kgexp.common import legacy_eval_module  # noqa: E402
from kgexp.fast_eval import (evaluate_binary_maps_fast,  # noqa: E402
                             evaluate_image_thresholds_fast, _greedy_by_order)


def _scene(seed, h=83, w=117, n_ann=3):
    rng = np.random.default_rng(seed)
    lab = np.zeros((h, w), np.int32)
    for k in range(1, 7):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        cv2.ellipse(lab, (int(cx), int(cy)), (int(rng.integers(8, 30)), int(rng.integers(8, 30))),
                    float(rng.uniform(0, 180)), 0, 360, k, -1)
    edge = (np.abs(np.diff(lab, axis=0, prepend=0)) + np.abs(np.diff(lab, axis=1, prepend=0))) > 0
    gts = []
    for a in range(n_ann):
        g = np.roll(edge, (a % 2, a // 2), axis=(0, 1)).copy()
        g[rng.random(g.shape) < 0.02] = True          # anotasi berisik
        gts.append(g)
    soft = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 1.5)
    pred = np.clip(soft / soft.max() + 0.35 * rng.random((h, w)), 0, 1)
    pred = (np.round(pred * 255) / 255).astype(np.float32)   # kuantisasi 8-bit seperti pipeline
    return pred, gts


def test_fast_equals_04c_thresholds():
    ev = legacy_eval_module()
    thr = np.linspace(0.01, 0.99, 99)
    for seed in range(4):
        pred, gts = _scene(seed)
        a = np.asarray(ev.evaluate_image_thresholds(pred, gts, thr, 0.0075), np.int64)
        b = evaluate_image_thresholds_fast(pred, gts, thr, 0.0075)
        assert np.array_equal(a, b), seed


def test_fast_equals_04c_hysteresis_and_large_radius():
    ev = legacy_eval_module()
    thr = np.linspace(0.05, 0.95, 19)
    for seed in range(3):
        pred, gts = _scene(10 + seed)
        maps = [apply_hysteresis_threshold(pred, 0.4 * t, t) for t in thr]
        h, w = pred.shape
        for frac in (0.0075, 0.03):                    # radius besar → banyak tie dan kandidat
            r = frac * np.sqrt(h * h + w * w)
            cache, tot = [], 0
            for g in gts:
                from skimage.morphology import thin
                from scipy.spatial import KDTree
                pts = np.argwhere(thin(g))
                cache.append((pts, KDTree(pts) if len(pts) else None))
                tot += len(pts)
            a = np.array([ev.evaluate_threshold_bsds(m, cache, tot, r) for m in maps], np.int64)
            b = evaluate_binary_maps_fast(maps, gts, frac)
            assert np.array_equal(a, b), (seed, frac)


def test_greedy_rounds_equal_sequential():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n_p, n_g, E = rng.integers(1, 30), rng.integers(1, 30), rng.integers(1, 200)
        rows, cols = rng.integers(0, n_p, E), rng.integers(0, n_g, E)
        cost = rng.integers(0, 5, E).astype(float)      # banyak tie
        order = np.argsort(cost)
        pm, gm = np.zeros(n_p, bool), np.zeros(n_g, bool)
        for i in order:
            if not pm[rows[i]] and not gm[cols[i]]:
                pm[rows[i]] = gm[cols[i]] = True
        pm2, ng2 = _greedy_by_order(rows, cols, cost, n_p, n_g)
        assert np.array_equal(pm, pm2) and int(gm.sum()) == ng2
