"""B2 — tuning setara untuk semua baseline di BSDS500 train+val (protokol sama dengan B1).

GWC   : grid k × λ yang sama dengan B1
Canny : Canny penuh (NMS + hysteresis, low = 0.4·high), σ ∈ grid; high di-sweep sebagai
        threshold ODS (menjawab C3/C4: kurva PR Canny berasal dari sweep threshold tinggi)
Sobel : σ pra-smoothing ∈ grid ; LoG: σ ∈ grid, ukuran kernel 2⌈3σ⌉+1
PC    : nscale × λ_min ; ED: gradient threshold × anchor threshold
Output: results/b2/grid_<metode>.csv, selected_config.json (GWC, Canny, Sobel, LoG, PC, ED),
        results/b2/test_tuned.csv (3 dataset), results/b2/decision.json (pelunakan klaim GWi > GWC)
"""
from __future__ import annotations

import itertools

import pandas as pd

from _cli import parser, setup, samples_for
from b1_tune_gwi import grid_specs, merge_splits
from kgexp.common import load_config, results_dir, write_json
from kgexp.evaluate import run_counts
from kgexp.methods import legacy_specs, tuned_specs
from kgexp.report import save_table, score_table
from kgexp.stats import score


def families(cfg):
    tu, pre = cfg["tuning"], cfg["preproc_default"]
    fam = {"GWC": [(s, {"ksize": s["ksize"], "wavelength": s["wavelength"]}) for s in grid_specs(cfg, "gwc")]}
    fam["Canny"] = [({"name": f"Canny_hyst_s{s:g}", "det": "canny_hyst", "pre": pre, "sigma": s,
                      "hyst_low_ratio": tu["canny_low_ratio"]}, {"sigma": s}) for s in tu["canny_sigma"]]
    fam["Sobel"] = [({"name": f"Sobel_s{s:g}", "det": "sobel", "pre": pre, "sigma": s}, {"sigma": s})
                    for s in tu["sobel_sigma"]]
    fam["LoG"] = [({"name": f"LoG_s{s:g}", "det": "log", "pre": pre, "sigma": s}, {"sigma": s})
                  for s in tu["log_sigma"]]
    fam["PC"] = [({"name": f"PC_n{n}_w{w}", "det": "pc", "pre": pre, "nscale": n, "min_wavelength": w},
                  {"nscale": n, "min_wavelength": w})
                 for n, w in itertools.product(tu["pc"]["nscale"], tu["pc"]["min_wavelength"])]
    fam["ED"] = [({"name": f"ED_g{g}_a{a}", "det": "ed", "pre": pre, "gradient_threshold": g, "anchor_threshold": a},
                  {"gradient_threshold": g, "anchor_threshold": a})
                 for g, a in itertools.product(tu["ed"]["gradient_threshold"], tu["ed"]["anchor_threshold"])]
    return fam


def main():
    p = parser(__doc__)
    p.add_argument("--only", nargs="+", default=None, help="subset keluarga, mis. GWC Canny")
    args = p.parse_args()
    cfg = setup(args)
    out = results_dir(cfg, "b2")
    try:
        import phasepack  # noqa: F401
        has_pc = True
    except ImportError:
        has_pc = False
    fam = families(cfg)
    sel = {}
    for name, entries in fam.items():
        if args.only and name not in args.only:
            continue
        if name == "PC" and not has_pc:
            print("[WARN] phasepack tidak terpasang → PC dilewati")
            continue
        specs = [e[0] for e in entries]
        parts = [run_counts(cfg, "BSDS500", sp, "clean", specs, samples_for(cfg, "BSDS500", sp, args.limit),
                            force=args.force) for sp in ("train", "val")]
        pooled = {s["name"]: merge_splits([parts[0][s["name"]], parts[1][s["name"]]]) for s in specs}
        df = score_table(pooled, cfg)
        save_table(df, out / f"grid_{name}")
        i = int(df.ods.idxmax())
        sel[name] = {**entries[i][1], "ods_trainval": float(df.ods.iloc[i])}
        print(f"[B2] {name}: {entries[i][1]}  ODS(train+val)={df.ods.iloc[i]:.4f}")

    selected = load_config(args.config)["selected"]
    selected.update(sel)
    write_json(cfg["paths"]["results"] / "selected_config.json", selected)
    cfg["selected"] = selected

    # Test: baseline tuned + default naskah, 3 dataset
    rows, decisions = [], {}
    for ds in ["BSDS500", "BIPED", "UDED"]:
        try:
            smp = samples_for(cfg, ds, "test", args.limit)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[WARN] {ds}: {e}")
            continue
        specs = tuned_specs(cfg) + [s for s in legacy_specs(cfg) if s["name"] == "GWC"]
        cs = run_counts(cfg, ds, "test", "clean", specs, smp, force=args.force)
        rows.append(score_table(cs, cfg, {"dataset": ds}))
        o = {k: score(v.counts, v.thresholds).ods for k, v in cs.items()}
        gap_def = o["GWi"] - o["GWC"]
        gap_tun = o["GWi"] - o["GWC_tuned"]
        soften = gap_def > 0 and gap_tun < (1 - cfg["gates"]["b1_gwc_soften_ratio"]) * gap_def
        decisions[ds] = {"gap_default": gap_def, "gap_tuned": gap_tun, "soften_claim_GWi_gt_GWC": bool(soften)}
    if rows:
        save_table(pd.concat(rows, ignore_index=True), out / "test_tuned")
    write_json(out / "decision.json", decisions)
    print(decisions)


if __name__ == "__main__":
    main()
