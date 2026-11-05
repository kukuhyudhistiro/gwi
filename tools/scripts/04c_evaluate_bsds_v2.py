"""
04c_evaluate_bsds_v2.py
Author: Kukuh Yudhistiro, 2026

 Per-annotator matching following Arbelaez et al. (2011).
                     Each predicted pixel is matched independently against
                     each annotator. A prediction is TP if matched by ANY
                     annotator. Total GT count = sum across annotators.

Algorithm per threshold t (single image):
    pred_bin = (pred >= t)
    pred_thin = thin(pred_bin)
    pred_pts = positions of pred_thin

    matched_pred = vector of bool, size = len(pred_pts), init False
    total_gt_count = 0
    matched_gt_count = 0
    matched_pred_count = 0

    For each annotator gt in gts:
        gt_thin = thin(gt)
        gt_pts = positions
        total_gt_count += len(gt_pts)

        # Bipartite matching: each gt pixel matched to at most 1 pred pixel
        # within max_dist; each pred pixel can match multiple annotators
        # (TP if matched by any annotator)
        matches = match_pixels(pred_pts, gt_pts, max_dist)
        matched_gt_count += matches.gt_matched_count
        matched_pred[matches.pred_indices] = True

    matched_pred_count = matched_pred.sum()

    Precision = matched_pred_count / len(pred_pts)
    Recall    = matched_gt_count / total_gt_count
    F         = 2PR / (P+R)

"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import KDTree as cKDTree
from skimage.morphology import thin as sk_thin


# ============================================================================
# GT loaders (same as 04b)
# ============================================================================

def load_gt_bsds_mat(mat_path: Path) -> List[np.ndarray]:
    """Load BSDS500 multi-annotator GT as list of bool arrays."""
    mat = loadmat(str(mat_path))
    gt = mat["groundTruth"]
    n_ann = gt.shape[1]
    gts = []
    for i in range(n_ann):
        b = gt[0, i]["Boundaries"][0, 0]
        gts.append((b > 0).astype(bool))
    return gts


def load_gt_png(png_path: Path) -> List[np.ndarray]:
    """Load single-annotator PNG GT, wrapped in a list."""
    arr = np.array(Image.open(str(png_path)).convert("L"))
    return [(arr > 127).astype(bool)]


def load_prediction_png(png_path: Path) -> np.ndarray:
    """Load 8-bit prediction PNG as float32 in [0,1]."""
    arr = np.array(Image.open(str(png_path)).convert("L"))
    return arr.astype(np.float32) / 255.0


# ============================================================================
# Single-annotator matching (greedy bipartite, identical to 04b semantics)
# ============================================================================

def match_single_annotator(pred_pts: np.ndarray, gt_pts: np.ndarray,
                            tree_gt, max_dist_px: float
                            ) -> Tuple[np.ndarray, int]:
    """Greedy bipartite matching between pred and one annotator's GT pts.

    Args:
        tree_gt : pre-built KDTree over gt_pts (None if gt_pts is empty)

    Returns:
        pred_matched_mask : bool array len(pred_pts), True if matched to any GT
        n_gt_matched      : number of GT points that got matched
    """
    n_pred = len(pred_pts)
    n_gt = len(gt_pts)
    pred_matched = np.zeros(n_pred, dtype=bool)
    if n_pred == 0 or n_gt == 0 or tree_gt is None:
        return pred_matched, 0

    # KDTree query for candidate matches (tree pre-built once per image)
    candidates = tree_gt.query_ball_point(pred_pts, r=max_dist_px)

    pred_idx_list, gt_idx_list, cost_list = [], [], []
    for p_i, cand in enumerate(candidates):
        for g_i in cand:
            d = np.linalg.norm(pred_pts[p_i] - gt_pts[g_i])
            pred_idx_list.append(p_i)
            gt_idx_list.append(g_i)
            cost_list.append(d)

    if not cost_list:
        return pred_matched, 0

    # Greedy match by ascending cost
    order = np.argsort(cost_list)
    gt_matched = np.zeros(n_gt, dtype=bool)
    for idx in order:
        p, g = pred_idx_list[idx], gt_idx_list[idx]
        if not pred_matched[p] and not gt_matched[g]:
            pred_matched[p] = True
            gt_matched[g] = True

    return pred_matched, int(gt_matched.sum())


# ============================================================================
# BSDS-correct multi-annotator evaluation per threshold
# ============================================================================

def evaluate_threshold_bsds(pred_bool: np.ndarray,
                            gt_cache: List[Tuple[np.ndarray, object]],
                            total_gt: int,
                            max_dist_px: float) -> Tuple[int, int, int, int]:
    """BSDS500-correct: per-annotator matching with prediction-side aggregation.

    For each annotator, match prediction pts against that annotator's GT.
    A prediction pixel is TP if matched by ANY annotator (union OR across
    annotator matchings). Total GT count is SUM across annotators.

    Args:
        pred_bool : binary prediction (threshold sudah diterapkan)
        gt_cache  : list of (gt_pts, tree_gt) pre-computed per image
        total_gt  : pre-summed total GT count across annotators

    Returns:
        count_r : true positives for recall = total matched GT (summed)
        sum_r   : total GT positives (summed across annotators)
        count_p : true positives for precision = unique matched pred pts
        sum_p   : total predicted positives
    """
    pred_thin = sk_thin(pred_bool)
    pred_pts = np.argwhere(pred_thin)
    n_pred = len(pred_pts)
    if n_pred == 0:
        return 0, total_gt, 0, 0

    pred_matched_any = np.zeros(n_pred, dtype=bool)
    total_gt_matched = 0

    for gt_pts, tree_gt in gt_cache:
        pred_matched, n_gt_matched = match_single_annotator(
            pred_pts, gt_pts, tree_gt, max_dist_px
        )
        pred_matched_any |= pred_matched
        total_gt_matched += n_gt_matched

    count_p = int(pred_matched_any.sum())
    count_r = int(total_gt_matched)
    return count_r, total_gt, count_p, n_pred


# ============================================================================
# Per-image multi-threshold evaluation
# ============================================================================

def evaluate_image_thresholds(pred: np.ndarray, gts: List[np.ndarray],
                              thresholds: np.ndarray,
                              max_dist_frac: float
                              ) -> List[Tuple[int, int, int, int]]:
    h, w = pred.shape
    diag = np.sqrt(h * h + w * w)
    max_dist_px = max_dist_frac * diag

    # Pre-compute thinning + KDTree per annotator (konstan untuk semua threshold).
    # Ini menghilangkan kerja yang dulu diulang n_thresholds kali per gambar.
    gt_cache: List[Tuple[np.ndarray, object]] = []
    total_gt = 0
    for gt in gts:
        gt_pts = np.argwhere(sk_thin(gt))
        tree_gt = cKDTree(gt_pts) if len(gt_pts) > 0 else None
        gt_cache.append((gt_pts, tree_gt))
        total_gt += len(gt_pts)

    results = []
    for thresh in thresholds:
        bmap = pred >= thresh
        counts = evaluate_threshold_bsds(bmap, gt_cache, total_gt, max_dist_px)
        results.append(counts)
    return results


# ============================================================================
# Aggregation
# ============================================================================

@dataclass
class DatasetScore:
    dataset: str
    method: str
    n_images: int
    n_thresholds: int
    ods_threshold: float
    ods_precision: float
    ods_recall: float
    ods_f: float
    ois_precision: float
    ois_recall: float
    ois_f: float
    ap: float


def aggregate_scores(per_image_counts, thresholds, dataset, method):
    n_images = len(per_image_counts)
    n_thresh = len(thresholds)
    eps = 1e-10

    ods_counts = np.zeros((n_thresh, 4))
    for img_counts in per_image_counts:
        for t_i, (cr, sr, cp, sp) in enumerate(img_counts):
            ods_counts[t_i] += [cr, sr, cp, sp]

    p_t = ods_counts[:, 2] / np.maximum(ods_counts[:, 3], eps)
    r_t = ods_counts[:, 0] / np.maximum(ods_counts[:, 1], eps)
    f_t = 2 * p_t * r_t / np.maximum(p_t + r_t, eps)
    ods_i = int(np.argmax(f_t))

    ois_p_list, ois_r_list, ois_f_list = [], [], []
    for img_counts in per_image_counts:
        a = np.array(img_counts, dtype=np.float64)
        p_i = a[:, 2] / np.maximum(a[:, 3], eps)
        r_i = a[:, 0] / np.maximum(a[:, 1], eps)
        f_i = 2 * p_i * r_i / np.maximum(p_i + r_i, eps)
        b = int(np.argmax(f_i))
        ois_p_list.append(p_i[b])
        ois_r_list.append(r_i[b])
        ois_f_list.append(f_i[b])

    ap_list = []
    for img_counts in per_image_counts:
        a = np.array(img_counts, dtype=np.float64)
        p_i = a[:, 2] / np.maximum(a[:, 3], eps)
        r_i = a[:, 0] / np.maximum(a[:, 1], eps)
        order = np.argsort(r_i)
        # Use trapezoid (np.trapz deprecated in NumPy 2.x)
        try:
            ap_list.append(float(np.trapezoid(p_i[order], r_i[order])))
        except AttributeError:
            ap_list.append(float(np.trapz(p_i[order], r_i[order])))

    return (DatasetScore(
        dataset=dataset, method=method, n_images=n_images,
        n_thresholds=n_thresh,
        ods_threshold=float(thresholds[ods_i]),
        ods_precision=float(p_t[ods_i]),
        ods_recall=float(r_t[ods_i]),
        ods_f=float(f_t[ods_i]),
        ois_precision=float(np.mean(ois_p_list)),
        ois_recall=float(np.mean(ois_r_list)),
        ois_f=float(np.mean(ois_f_list)),
        ap=float(np.mean(ap_list)),
    ), pd.DataFrame({
        "threshold": thresholds,
        "precision": p_t,
        "recall": r_t,
        "f_measure": f_t,
    }))


# ============================================================================
# Per-(method, dataset) driver
# ============================================================================

def evaluate_method_dataset(pred_dir: Path, gt_dir: Path,
                            dataset: str, method: str,
                            max_dist_frac: float,
                            n_thresholds: int,
                            gt_format: str):
    thresholds = np.linspace(
        1.0 / (n_thresholds + 1),
        1.0 - 1.0 / (n_thresholds + 1),
        n_thresholds,
    )
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions in {pred_dir}")

    per_image_counts = []
    n_proc, n_skip = 0, 0
    n_annotator_total = 0
    for pred_path in pred_files:
        stem = pred_path.stem
        gt_path = (gt_dir / f"{stem}.mat" if gt_format == "mat"
                   else gt_dir / f"{stem}.png")
        if not gt_path.exists():
            n_skip += 1
            continue
        try:
            gts = (load_gt_bsds_mat(gt_path) if gt_format == "mat"
                   else load_gt_png(gt_path))
            n_annotator_total += len(gts)
            pred = load_prediction_png(pred_path)
            counts = evaluate_image_thresholds(
                pred, gts, thresholds, max_dist_frac
            )
            per_image_counts.append(counts)
            n_proc += 1
            if n_proc % 10 == 0:
                print(f"  [{dataset}/{method}] {n_proc}/{len(pred_files)}")
        except Exception as e:
            print(f"  [ERR] {stem}: {e}")
            n_skip += 1

    avg_ann = n_annotator_total / max(n_proc, 1)
    print(f"  Processed: {n_proc}, skipped: {n_skip}, avg annotators/img: {avg_ann:.2f}")
    if n_proc == 0:
        raise RuntimeError(f"No images evaluated for {dataset}/{method}")
    return aggregate_scores(per_image_counts, thresholds, dataset, method)


# ============================================================================
# CLI
# ============================================================================

DATASET_LAYOUTS = {
    "BSDS500": {"gt_subdir": "groundTruth/test", "gt_format": "mat"},
    "BIPED": {"gt_subdir": "edge_maps/test", "gt_format": "png"},
    "UDED": {"gt_subdir": "gt", "gt_format": "png"},
}


def main():
    parser = argparse.ArgumentParser(
        description="BSDS500-correct ODS/OIS/AP evaluation (per-annotator matching)."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+",
                        default=["BSDS500", "BIPED", "UDED"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--n-thresholds", type=int, default=99)
    parser.add_argument("--max-dist", type=float, default=0.0075,
                        help="Tolerance as fraction of diagonal (default 0.0075)")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    all_scores = []

    for dataset in args.datasets:
        if dataset not in DATASET_LAYOUTS:
            print(f"[WARN] Unknown dataset {dataset}")
            continue
        layout = DATASET_LAYOUTS[dataset]
        gt_dir = args.data_root / dataset / layout["gt_subdir"]
        pred_root = args.output_root / dataset
        if not gt_dir.exists() or not pred_root.exists():
            print(f"[WARN] Skipping {dataset}: paths missing")
            continue

        if args.methods is None:
            method_dirs = sorted([d for d in pred_root.iterdir()
                                  if d.is_dir() and not d.name.endswith("_binary")])
        else:
            method_dirs = [pred_root / m for m in args.methods
                           if (pred_root / m).exists()]

        print(f"\n{'=' * 70}\n  Dataset: {dataset}\n{'=' * 70}")
        for method_dir in method_dirs:
            method = method_dir.name
            print(f"\n[INFO] Evaluating {dataset}/{method}")
            try:
                score, pr_df = evaluate_method_dataset(
                    pred_dir=method_dir, gt_dir=gt_dir,
                    dataset=dataset, method=method,
                    max_dist_frac=args.max_dist,
                    n_thresholds=args.n_thresholds,
                    gt_format=layout["gt_format"],
                )
                all_scores.append(score)
                pr_df.to_csv(
                    args.results_dir / f"pr_curve_{dataset}_{method}.csv",
                    index=False,
                )
                print(f"  ODS = {score.ods_f:.4f}  "
                      f"OIS = {score.ois_f:.4f}  "
                      f"AP = {score.ap:.4f}")
            except Exception as e:
                print(f"  [FAIL] {e}")

    if all_scores:
        df = pd.DataFrame([asdict(s) for s in all_scores])
        summary_path = args.results_dir / "ods_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
        print(df[["dataset", "method", "n_images",
                  "ods_f", "ois_f", "ap"]].to_string(index=False))
        print(f"\n[OK] Summary: {summary_path}")


if __name__ == "__main__":
    main()
