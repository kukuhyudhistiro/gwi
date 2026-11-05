"""B5 — kalibrasi protokol evaluasi.

Untuk setiap entri b5.references di config (metode + dataset + ODS yang dilaporkan di
paper aslinya, diisi manual dari sumber resmi), peta keluaran resmi metode itu
(results/maps/<metode>/<dataset>/) dievaluasi dengan:
    greedy  — evaluator naskah (04c)
    maxcard — maximum-cardinality bipartite matching dalam radius yang sama
Offset protokol = ODS_reported − ODS_greedy. Selain itu dilaporkan selisih greedy vs maxcard
untuk SEMUA metode klasik di BSDS500 test (efek aturan matching saja).

Catatan: maxcard lebih dekat ke correspondPixels BSDS (assignment), tetapi belum identik
(BSDS memakai biaya jarak + outlier cost dan toleransi 0.0075 pada ukuran asli). Untuk
angka resmi, bandingkan juga dengan benchmark MATLAB atau implementasi Python yang
dipakai di literatur terkini.
Keputusan: offset > gates.b5_offset_max → pertimbangkan ganti evaluator utama dan rerun M9.
"""
from __future__ import annotations

import pandas as pd

from _cli import parser, setup, samples_for
from kgexp.common import results_dir, write_json
from kgexp.evaluate import counts_from_maps, run_counts
from kgexp.methods import tuned_specs
from kgexp.report import save_table
from kgexp.stats import score


def main():
    p = parser(__doc__)
    p.add_argument("--classical", action="store_true", help="juga ukur greedy vs maxcard untuk metode klasik")
    args = p.parse_args()
    cfg = setup(args)
    out = results_dir(cfg, "b5")
    rows = []
    for ref in cfg["b5"]["references"]:
        d = cfg["paths"]["results"] / "maps" / ref["method"] / ref["dataset"]
        if not d.exists():
            print(f"[WARN] {d} tidak ada")
            continue
        r = {"method": ref["method"], "dataset": ref["dataset"], "reported_ods": ref.get("reported_ods"),
             "source": ref.get("source")}
        for m in ("greedy", "maxcard"):
            cs = counts_from_maps(cfg, ref["dataset"], "test", d, f"{ref['method']}_{m}", match=m)
            r[f"ods_{m}"] = score(cs.counts, cs.thresholds).ods
        if r["reported_ods"] is not None:
            r["offset_greedy"] = r["reported_ods"] - r["ods_greedy"]
            r["offset_maxcard"] = r["reported_ods"] - r["ods_maxcard"]
        rows.append(r)
    if args.classical:
        smp = samples_for(cfg, "BSDS500", "test", args.limit)
        specs = tuned_specs(cfg)
        g = run_counts(cfg, "BSDS500", "test", "clean", specs, smp, match="greedy", force=args.force)
        m = run_counts(cfg, "BSDS500", "test", "clean", specs, smp, match="maxcard", force=args.force)
        for s in specs:
            n = s["name"]
            rows.append({"method": n, "dataset": "BSDS500", "reported_ods": None,
                         "ods_greedy": score(g[n].counts, g[n].thresholds).ods,
                         "ods_maxcard": score(m[n].counts, m[n].thresholds).ods})
    if not rows:
        print("[INFO] b5.references kosong dan --classical tidak dipakai; tidak ada yang dihitung.")
        return
    df = pd.DataFrame(rows)
    df["greedy_minus_maxcard"] = df.ods_greedy - df.ods_maxcard
    save_table(df, out / "calibration")
    off = df.get("offset_greedy")
    worst = float(off.abs().max()) if off is not None and off.notna().any() else None
    lim = cfg["gates"]["b5_offset_max"]
    write_json(out / "decision.json", {
        "max_abs_offset": worst, "threshold": lim,
        "decision": None if worst is None else ("GANTI evaluator + rerun M9" if worst > lim
                                                else "laporkan sebagai keterbatasan")})
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
