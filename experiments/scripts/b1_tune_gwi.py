"""B1 — tuning GWi dan GWi+ODPS di BSDS500 train+val; test hanya untuk pelaporan.

Grid  : k × λ (config.tuning.gabor_grid), σ = 0.56·λ, γ = 0.5, N = 8, d_ODPS = floor(λ/2+0.5).
Pilih : ODS tertinggi di train+val; kalau selisih dengan yang terbaik < tie_eps → k terkecil,
        lalu λ terkecil.
Cek   : stabilitas di BIPED train (peringkat konfigurasi terpilih), bukan untuk seleksi.
Output: results/b1/grid_{bsds_trainval,biped_train}.csv, results/selected_config.json (GWi, GWi_odps),
        results/b1/test_row.csv (satu baris test untuk konfigurasi terpilih).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from _cli import parser, setup, samples_for
from kgexp.common import load_config, results_dir, write_json
from kgexp.evaluate import CountSet, run_counts
from kgexp.report import save_table, score_table
from kgexp.stats import score


def grid_specs(cfg, det):
    gg = cfg["tuning"]["gabor_grid"]
    md = cfg["methods_default"]
    out = []
    for k, lam in itertools.product(gg["ksize"], gg["wavelength"]):
        s = {"name": f"{det}_k{k}_l{lam:g}", "det": det, "pre": cfg["preproc_default"], "ksize": k,
             "wavelength": float(lam), "gamma": md["gabor"]["gamma"], "n_orientations": md["gabor"]["n_orientations"]}
        if "border" in md["gabor"]:
            s["border"] = md["gabor"]["border"]
        if det == "gwi_odps":
            s["odps_d"] = md["odps_d"]
        out.append(s)
    return out


def merge_splits(parts):
    """Gabungkan count set beberapa split (train+val) jadi satu pool citra."""
    a = parts[0]
    return CountSet(a.name, np.concatenate([p.image_ids for p in parts]), a.thresholds,
                    np.concatenate([p.counts for p in parts]), None, a.spec)


def select(df, tie_eps):
    best = df.ods.max()
    cand = df[df.ods > best - tie_eps].sort_values(["ksize", "wavelength"])
    return cand.iloc[0]


def tune_family(cfg, specs, splits, limit, force):
    res = {}
    for ds, sp in splits:
        res[sp] = run_counts(cfg, ds, sp, "clean", specs, samples_for(cfg, ds, sp, limit), force=force)
    pooled = {s["name"]: merge_splits([res[sp][s["name"]] for _, sp in splits]) for s in specs}
    df = score_table(pooled, cfg)
    df["ksize"] = [s["ksize"] for s in specs]
    df["wavelength"] = [s["wavelength"] for s in specs]
    return df


def main():
    p = parser(__doc__)
    p.add_argument("--skip-biped", action="store_true")
    args = p.parse_args()
    cfg = setup(args)
    out = results_dir(cfg, "b1")
    tie = cfg["tuning"]["tie_eps"]
    selected, rows_test = {}, []
    for det, key in [("gwi", "GWi"), ("gwi_odps", "GWi_odps")]:
        specs = grid_specs(cfg, det)
        df = tune_family(cfg, specs, [("BSDS500", "train"), ("BSDS500", "val")], args.limit, args.force)
        df["family"] = key
        save_table(df, out / f"grid_bsds_trainval_{key}")
        s = select(df, tie)
        selected[key] = {"ksize": int(s.ksize), "wavelength": float(s.wavelength), "ods_trainval": float(s.ods)}
        print(f"[B1] {key}: k={int(s.ksize)}, λ={s.wavelength:g}, ODS(train+val)={s.ods:.4f}")

        if not args.skip_biped:
            try:
                db = tune_family(cfg, specs, [("BIPED", "train")], args.limit, args.force)
                db["rank"] = db.ods.rank(ascending=False, method="min").astype(int)
                save_table(db, out / f"grid_biped_train_{key}")
                r = db[(db.ksize == s.ksize) & (db.wavelength == s.wavelength)]["rank"].iloc[0]
                sb = select(db, tie)
                selected[key]["biped_rank"] = int(r)
                selected[key]["biped_same_choice"] = bool(sb.ksize == s.ksize and sb.wavelength == s.wavelength)
            except (FileNotFoundError, KeyError, RuntimeError) as e:
                print(f"[WARN] BIPED train dilewati: {e}")

        spec = next(x for x in specs if x["ksize"] == s.ksize and x["wavelength"] == s.wavelength)
        spec = {**spec, "name": key}
        cs = run_counts(cfg, "BSDS500", "test", "clean", [spec], samples_for(cfg, "BSDS500", "test", args.limit),
                        force=args.force)
        rows_test.append(score_table(cs, cfg, {"dataset": "BSDS500", "split": "test"}))

    save_table(pd.concat(rows_test, ignore_index=True), out / "test_row")
    g0 = cfg["methods_default"]["gabor"]
    k0, l0 = int(g0["ksize"]), float(g0["wavelength"])
    differs = any(v["ksize"] != k0 or v["wavelength"] != l0 for v in selected.values())
    if cfg["tuning"].get("gabor_select", True):
        sel_path = cfg["paths"]["results"] / "selected_config.json"
        prev = load_config(args.config)["selected"]
        prev.update(selected)
        write_json(sel_path, prev)
        write_json(out / "decision.json", {"selected": selected, "rerun_all_tables_in_M9": differs})
        print("[B1] Keputusan:", "konfigurasi BERUBAH → semua tabel dihitung ulang di M9" if differs
              else f"tetap (k={k0}, λ={l0:g})")
    else:
        # GWi dikunci ke konfigurasi Paper 2; B1 hanya sensitivitas, selected_config.json tidak disentuh.
        write_json(out / "decision.json", {"mode": "sensitivity_only", "fixed": {"ksize": k0, "wavelength": l0},
                                           "best_in_grid": selected, "best_differs_from_fixed": differs})
        print(f"[B1] Mode sensitivitas: GWi tetap k={k0}, λ={l0:g} (Paper 2). Terbaik di grid:",
              ", ".join(f"{k} k={v['ksize']} λ={v['wavelength']:g}" for k, v in selected.items()))


if __name__ == "__main__":
    main()
