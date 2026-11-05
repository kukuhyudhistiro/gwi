"""M0b — bangun cache count untuk 8 metode naskah (konfigurasi default k=5, λ=4) dan
uji regresi terhadap eval_results/k5/ods_summary.csv sampai 4 desimal.

Contoh:
    python scripts/m0_cache.py                      # semua dataset test
    python scripts/m0_cache.py --datasets BSDS500 --methods GWi GWi_odps Canny
"""
from __future__ import annotations

import sys

import pandas as pd

from _cli import parser, setup, samples_for
from kgexp.common import results_dir
from kgexp.evaluate import run_counts
from kgexp.methods import legacy_specs
from kgexp.report import save_table, score_table


def main():
    p = parser(__doc__)
    p.add_argument("--datasets", nargs="+", default=["BSDS500", "BIPED", "UDED"])
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--tol", type=float, default=5e-5, help="toleransi regresi (|Δ| < 0.5e-4 = sama s.d. 4 desimal)")
    args = p.parse_args()
    cfg = setup(args)
    specs = [s for s in legacy_specs(cfg) if not args.methods or s["name"] in args.methods]
    try:
        import phasepack  # noqa: F401
    except ImportError:
        specs = [s for s in specs if s["det"] != "pc"]
        print("[WARN] phasepack tidak terpasang → PC dilewati")
    tables = []
    for ds in args.datasets:
        cs = run_counts(cfg, ds, "test", "clean", specs, samples_for(cfg, ds, "test", args.limit),
                        force=args.force)
        tables.append(score_table(cs, cfg, {"dataset": ds}))
    df = pd.concat(tables, ignore_index=True)
    out = results_dir(cfg, "m0")
    save_table(df, out / "ods_from_cache")

    legacy = cfg["paths"]["legacy_summary"]
    if not legacy.exists() or args.limit:
        print("[INFO] uji regresi dilewati (ods_summary.csv tidak ada atau --limit dipakai)")
        print(df.round(4).to_string(index=False))
        return
    ref = pd.read_csv(legacy)
    m = df.merge(ref, on=["dataset", "method"], suffixes=("", "_ref"))
    m["d_ods"] = m["ods"] - m["ods_f"]
    m["d_ois"] = m["ois"] - m["ois_f"]
    m["d_ap"] = m["ap"] - m["ap_ref"]
    m["pass"] = (m[["d_ods", "d_ois", "d_ap"]].abs() < args.tol).all(axis=1)
    save_table(m[["dataset", "method", "ods", "ods_f", "d_ods", "d_ois", "d_ap", "pass"]],
               out / "regression")
    print(m[["dataset", "method", "ods", "ods_f", "d_ods", "pass"]].round(5).to_string(index=False))
    if not m["pass"].all():
        print("[FAIL] Cache tidak mereproduksi Tabel 5–7. Jangan lanjut ke B1 sebelum selisihnya dijelaskan.")
        sys.exit(1)
    print("[OK] Regresi lulus: cache identik dengan Tabel 5–7 (4 desimal).")


if __name__ == "__main__":
    main()
