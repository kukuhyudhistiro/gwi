"""M0c — verifikasi evaluator cepat (engine=fast) terhadap 04c pada data asli.

Untuk n citra per dataset dan setiap metode naskah (+ Canny hysteresis), count per threshold
(TP_gt, N_gt, TP_pred, N_pred) dari kedua engine harus IDENTIK (selisih 0 di semua sel).
Jalankan sekali di M0 sebelum memakai engine=fast; lampirkan hasilnya di repo.

    python scripts/m0_verify_fast_eval.py --n 5
Output: results/m0/fast_eval_equivalence.csv ; exit 1 kalau ada selisih.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from _cli import parser, setup
from kgexp.common import results_dir
from kgexp.data import list_samples, load_gts, load_rgb, subset
from kgexp.evaluate import eval_image, thresholds_for
from kgexp.methods import build_pipeline, legacy_specs


def main():
    p = parser(__doc__)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--datasets", nargs="+", default=["BSDS500", "BIPED", "UDED"])
    p.add_argument("--methods", nargs="+", default=None)
    args = p.parse_args()
    cfg = setup(args)
    thr = thresholds_for(cfg["eval"]["n_thresholds"])
    md = cfg["eval"]["max_dist"]
    specs = legacy_specs(cfg) + [{"name": "Canny_hyst", "det": "canny_hyst", "pre": "he", "sigma": 1.4,
                                  "hyst_low_ratio": cfg["tuning"]["canny_low_ratio"]}]
    if args.methods:
        specs = [s for s in specs if s["name"] in args.methods]
    rows = []
    for ds in args.datasets:
        try:
            smp = subset(list_samples(cfg, ds, "test"), args.n, cfg["seed"])
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            print(f"[skip] {ds}: {e}")
            continue
        for s in smp:
            rgb, gts = load_rgb(s), load_gts(s)
            for sp in specs:
                try:
                    pred, _ = build_pipeline(sp, cfg)(rgb)
                except RuntimeError as e:          # mis. ED tanpa opencv-contrib
                    print(f"[skip] {sp['name']}: {e}")
                    continue
                hl = sp.get("hyst_low_ratio")
                t0 = time.perf_counter()
                cf = eval_image(pred, gts, thr, md, "greedy", hl, engine="fast")
                t1 = time.perf_counter()
                cl = eval_image(pred, gts, thr, md, "greedy", hl, engine="legacy")
                t2 = time.perf_counter()
                diff = int(np.abs(cf - cl).max())
                rows.append({"dataset": ds, "image_id": s.image_id, "method": sp["name"],
                             "max_abs_diff": diff, "fast_s": t1 - t0, "legacy_s": t2 - t1})
                print(f"{ds:8s} {s.image_id:>10s} {sp['name']:12s} diff={diff} "
                      f"fast={t1 - t0:.2f}s legacy={t2 - t1:.2f}s", flush=True)
    df = pd.DataFrame(rows)
    out = results_dir(cfg, "m0")
    df.to_csv(out / "fast_eval_equivalence.csv", index=False)
    ok = bool(len(df)) and int(df.max_abs_diff.max()) == 0
    print(f"\n[{'OK' if ok else 'FAIL'}] {len(df)} pasangan citra×metode; "
          f"selisih maksimum = {df.max_abs_diff.max() if len(df) else 'n/a'}; "
          f"speedup median = {(df.legacy_s / df.fast_s).median():.1f}×")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
