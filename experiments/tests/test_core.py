"""Uji unit: jalankan `python -m pytest tests -q` dari folder experiments/."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kgexp.common import legacy_eval_module  # noqa: E402
from kgexp.degrade import add_gaussian, simulate_lowlight  # noqa: E402
from kgexp.methods import gradient_nms, matched_canny_sigma  # noqa: E402
from kgexp.odps_fast import odps_fast  # noqa: E402
from kgexp.stats import holm, paired_delta, score  # noqa: E402
import nms_odps  # noqa: E402


def test_odps_fast_identical():
    rng = np.random.default_rng(0)
    for d in (1, 2, 3):
        mag = rng.random((97, 131)).astype(np.float64)
        oi = rng.integers(0, 8, mag.shape).astype(np.int32)
        a = nms_odps.odps(mag, oi, 8, d)
        b = odps_fast(mag, oi, 8, d)
        assert np.array_equal(a, b)


def test_score_matches_04c():
    rng = np.random.default_rng(1)
    thr = np.linspace(0.01, 0.99, 99)
    counts = []
    for _ in range(12):
        sp = rng.integers(50, 500, 99)
        cp = (sp * rng.uniform(0.2, 0.9, 99)).astype(int)
        sr = np.full(99, rng.integers(200, 600))
        cr = (sr * rng.uniform(0.2, 0.9, 99)).astype(int)
        counts.append(list(zip(cr, sr, cp, sp)))
    ref, _ = legacy_eval_module().aggregate_scores(counts, thr, "X", "Y")
    s = score(np.array(counts), thr)
    assert abs(s.ods - ref.ods_f) < 1e-12 and abs(s.ois - ref.ois_f) < 1e-12 and abs(s.ap - ref.ap) < 1e-12


def test_paired_delta_zero_for_identical():
    rng = np.random.default_rng(2)
    c = rng.integers(1, 100, (20, 99, 4))
    r = paired_delta(c, c, np.linspace(0.01, 0.99, 99), n_boot=200)
    assert r["delta"] == 0 and r["ci_lo"] == 0 and r["ci_hi"] == 0


def test_holm_monotone():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj[0] == 0.04 and adj[2] == 0.09 and adj[1] == 0.09 and adj[3] == 0.5


def test_nms_diagonal_direction():
    # tepi diagonal: intensitas naik ke arah kanan-bawah → gradien 45° (y ke bawah)
    n = 41
    yy, xx = np.mgrid[:n, :n]
    ridge = np.exp(-((xx + yy - n) ** 2) / 4.0)          # punggungan magnitude sepanjang anti-diagonal
    gx = gy = np.ones((n, n))
    out = gradient_nms(ridge, gx, gy)
    kept = np.argwhere(out[5:-5, 5:-5] > 0)
    assert len(kept) > 0
    assert np.all(np.abs(kept[:, 0] + 5 + kept[:, 1] + 5 - n) <= 1)


def test_lowlight_iso_preserves_mean():
    rng = np.random.default_rng(3)
    img = np.full((64, 64, 3), 128, np.uint8)
    y = simulate_lowlight(img, 0.05, rng, mode="iso")
    assert abs(y.mean() - 128) < 6
    yd = simulate_lowlight(img, 0.05, np.random.default_rng(3), mode="dark")
    assert yd.mean() < 60


def test_gaussian_zero_is_identity():
    img = np.random.default_rng(4).integers(0, 255, (8, 8, 3)).astype(np.uint8)
    assert add_gaussian(img, 0, None) is img


def test_matched_sigma_reasonable():
    s = matched_canny_sigma(5, 4.0)
    assert 0.3 <= s <= 2.0
