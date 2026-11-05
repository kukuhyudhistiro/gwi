"""Cache count per citra per threshold (M0) dan runner paralel.

Setiap hasil (dataset, split, kondisi, metode) disimpan sebagai
    results/counts/<dataset>/<split>/<kondisi>/<metode>__<hash>.npz
berisi image_ids, thresholds, counts[n_img, n_thr, 4] = (TP_gt, N_gt, TP_pred, N_pred)
— urutan kolom sama dengan 04c (count_r, sum_r, count_p, sum_p). Semua statistik
(ODS/OIS/AP, bootstrap, Wilcoxon, transfer threshold) dihitung dari cache ini
tanpa evaluasi ulang.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import thin as sk_thin

from .common import legacy_eval_module, results_dir, stable_hash
from .data import Sample, list_samples, load_gts, load_rgb
from . import degrade
from .fast_eval import evaluate_binary_maps_fast, evaluate_image_thresholds_fast
from .methods import build_pipeline


def thresholds_for(n: int) -> np.ndarray:
    return np.linspace(1.0 / (n + 1), 1.0 - 1.0 / (n + 1), n)


# =============================================================================
# Matching
# =============================================================================

def _gt_cache(gts, max_dist_px):
    cache, total = [], 0
    for gt in gts:
        pts = np.argwhere(sk_thin(gt))
        cache.append((pts, cKDTree(pts) if len(pts) else None))
        total += len(pts)
    return cache, total


def _match_maxcard(pred_pts, gt_pts, tree, r):
    """Maximum-cardinality bipartite matching dalam radius r (B5, mendekati
    correspondPixels BSDS yang memakai assignment, bukan greedy)."""
    n_p, n_g = len(pred_pts), len(gt_pts)
    if n_p == 0 or n_g == 0 or tree is None:
        return np.zeros(n_p, bool), 0
    cand = tree.query_ball_point(pred_pts, r=r)
    rows = np.repeat(np.arange(n_p), [len(c) for c in cand])
    cols = np.fromiter((g for c in cand for g in c), dtype=np.int64, count=len(rows))
    if len(rows) == 0:
        return np.zeros(n_p, bool), 0
    A = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_p, n_g))
    m = maximum_bipartite_matching(A, perm_type="column")   # m[i] = gt utk pred i, -1 jika tidak
    matched = m >= 0
    return matched, int(matched.sum())


def _counts_binary(pred_bool, gt_cache, total_gt, r, match):
    ev = legacy_eval_module()
    if match == "greedy":
        return ev.evaluate_threshold_bsds(pred_bool, gt_cache, total_gt, r)
    pts = np.argwhere(sk_thin(pred_bool))
    if len(pts) == 0:
        return 0, total_gt, 0, 0
    any_m = np.zeros(len(pts), bool)
    tp_gt = 0
    for gpts, tree in gt_cache:
        mm, n = _match_maxcard(pts, gpts, tree, r)
        any_m |= mm
        tp_gt += n
    return tp_gt, total_gt, int(any_m.sum()), len(pts)


def eval_image(pred: np.ndarray, gts: List[np.ndarray], thresholds: np.ndarray,
               max_dist_frac: float, match="greedy", hyst_low_ratio=None, engine="fast") -> np.ndarray:
    """Count per threshold. engine='fast' identik dengan 04c (lihat fast_eval.py), median 6.4× lebih cepat (2–20×)."""
    h, w = pred.shape
    r = max_dist_frac * np.sqrt(h * h + w * w)
    if match == "greedy" and engine == "fast":
        if hyst_low_ratio is None:
            return evaluate_image_thresholds_fast(pred, gts, thresholds, max_dist_frac)
        maps = [apply_hysteresis_threshold(pred, hyst_low_ratio * t, t) for t in thresholds]
        return evaluate_binary_maps_fast(maps, gts, max_dist_frac)
    if match == "greedy" and hyst_low_ratio is None:
        res = legacy_eval_module().evaluate_image_thresholds(pred, gts, thresholds, max_dist_frac)
        return np.asarray(res, dtype=np.int64)
    gt_cache, total = _gt_cache(gts, r)
    out = np.zeros((len(thresholds), 4), np.int64)
    for i, t in enumerate(thresholds):
        if hyst_low_ratio is None:
            b = pred >= t
        else:
            b = apply_hysteresis_threshold(pred, hyst_low_ratio * t, t)
        out[i] = _counts_binary(b, gt_cache, total, r, match)
    return out


# =============================================================================
# Cache
# =============================================================================

@dataclass
class CountSet:
    name: str
    image_ids: np.ndarray
    thresholds: np.ndarray
    counts: np.ndarray          # (n_img, n_thr, 4)
    runtime: Optional[np.ndarray] = None   # (n_img,) total_s (hanya indikatif; bukan harness timing)
    spec: Optional[dict] = None


def cache_path(cfg, dataset, split, condition, spec, match=None) -> Path:
    e = cfg["eval"]
    key = {"spec": spec, "n_thr": e["n_thresholds"], "max_dist": e["max_dist"],
           "match": match or e["match"], "deg": cfg["degradation"], "seed": cfg["seed"]}
    d = results_dir(cfg, "counts", dataset, split, condition)
    return d / f"{spec['name']}__{stable_hash(key)}.npz"


def save_countset(path: Path, cs: CountSet):
    np.savez_compressed(path, image_ids=cs.image_ids, thresholds=cs.thresholds,
                        counts=cs.counts, runtime=cs.runtime if cs.runtime is not None else np.array([]),
                        spec=json.dumps(cs.spec))


def load_countset(path: Path) -> CountSet:
    z = np.load(path, allow_pickle=False)
    spec = json.loads(str(z["spec"]))
    rt = z["runtime"]
    return CountSet(spec["name"], z["image_ids"], z["thresholds"], z["counts"],
                    rt if rt.size else None, spec)


# =============================================================================
# Runner paralel: satu worker memproses satu citra untuk semua metode
# =============================================================================

_W = {}


def _init_worker(cfg, specs, condition, match, save_maps):
    import cv2
    cv2.setNumThreads(1)
    _W.update(cfg=cfg, specs=specs, cond=degrade.Condition.parse(condition),
              pipes=[build_pipeline(s, cfg) for s in specs], match=match,
              thr=thresholds_for(cfg["eval"]["n_thresholds"]), save_maps=save_maps)


def _work(sample: Sample):
    cfg = _W["cfg"]
    rgb = degrade.apply(load_rgb(sample), _W["cond"], cfg, sample.image_id)
    gts = load_gts(sample)
    out = []
    for spec, pipe in zip(_W["specs"], _W["pipes"]):
        pred, tm = pipe(rgb)
        if _W["save_maps"]:
            p = Path(_W["save_maps"]) / spec["name"] / f"{sample.image_id}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.round(pred * 255).astype(np.uint8)).save(p)
        c = eval_image(pred, gts, _W["thr"], cfg["eval"]["max_dist"], _W["match"],
                       spec.get("hyst_low_ratio"), cfg["eval"].get("engine", "fast"))
        out.append((c, tm["total_s"]))
    return sample.image_id, out


def run_counts(cfg, dataset: str, split: str, condition: str, specs: List[dict],
               samples: Optional[List[Sample]] = None, n_jobs: Optional[int] = None,
               match: Optional[str] = None, save_maps: Optional[Path] = None,
               force: bool = False, verbose: bool = True) -> Dict[str, CountSet]:
    match = match or cfg["eval"]["match"]
    samples = samples if samples is not None else list_samples(cfg, dataset, split)
    ids = np.array(sorted(s.image_id for s in samples))
    samples = sorted(samples, key=lambda s: s.image_id)
    result, todo = {}, []
    for s in specs:
        p = cache_path(cfg, dataset, split, condition, s, match)
        if p.exists() and not force:
            cs = load_countset(p)
            if np.array_equal(cs.image_ids, ids):
                result[s["name"]] = cs
                continue
        todo.append(s)
    if todo:
        if verbose:
            print(f"[eval] {dataset}/{split}/{condition}: {len(samples)} citra × "
                  f"{[s['name'] for s in todo]}")
        n_jobs = n_jobs or cfg["eval"]["n_jobs"]
        n_jobs = max(1, min(n_jobs, os.cpu_count() or 1, len(samples)))
        thr = thresholds_for(cfg["eval"]["n_thresholds"])
        args = (cfg, todo, condition, match, str(save_maps) if save_maps else None)
        rows = {}
        if n_jobs == 1:
            _init_worker(*args)
            it = map(_work, samples)
            for k, (iid, out) in enumerate(it, 1):
                rows[iid] = out
                if verbose and k % 25 == 0:
                    print(f"   {k}/{len(samples)}")
        else:
            with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=args) as ex:
                for k, (iid, out) in enumerate(ex.map(_work, samples, chunksize=2), 1):
                    rows[iid] = out
                    if verbose and k % 25 == 0:
                        print(f"   {k}/{len(samples)}")
        for j, s in enumerate(todo):
            counts = np.stack([rows[i][j][0] for i in ids])
            rt = np.array([rows[i][j][1] for i in ids])
            cs = CountSet(s["name"], ids, thr, counts, rt, s)
            save_countset(cache_path(cfg, dataset, split, condition, s, match), cs)
            result[s["name"]] = cs
    return result


def counts_from_maps(cfg, dataset, split, maps_dir: Path, name: str,
                     match: Optional[str] = None, cache=True) -> CountSet:
    """Evaluasi peta probabilitas eksternal (B4 DL, B5 kalibrasi) dengan evaluator yang sama.
    Peta harus PNG 8-bit, nama file = image_id, resolusi = GT."""
    match = match or cfg["eval"]["match"]
    spec = {"name": name, "external_maps": str(maps_dir)}
    p = cache_path(cfg, dataset, split, "clean", spec, match)
    if cache and p.exists():
        return load_countset(p)
    thr = thresholds_for(cfg["eval"]["n_thresholds"])
    samples = sorted(list_samples(cfg, dataset, split), key=lambda s: s.image_id)
    ids, rows = [], []
    for s in samples:
        f = Path(maps_dir) / f"{s.image_id}.png"
        if not f.exists():
            continue
        pred = np.array(Image.open(f).convert("L")).astype(np.float32) / 255.0
        gts = load_gts(s)
        if pred.shape != gts[0].shape:
            raise ValueError(f"{f}: ukuran {pred.shape} ≠ GT {gts[0].shape}")
        rows.append(eval_image(pred, gts, thr, cfg["eval"]["max_dist"], match,
                               engine=cfg["eval"].get("engine", "fast")))
        ids.append(s.image_id)
    if len(ids) < len(samples):
        print(f"[WARN] {name}: {len(samples) - len(ids)} peta tidak ditemukan di {maps_dir}")
    cs = CountSet(name, np.array(ids), thr, np.stack(rows), None, spec)
    save_countset(p, cs)
    return cs
