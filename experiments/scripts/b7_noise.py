"""B7 / B7b — uji noise Gaussian dan low-light pada test set, parameter final tanpa tuning ulang.

B7  : BSDS500 test, noise Gaussian σ ∈ degradation.gaussian_sigmas
B7b : BSDS500 test, low-light (mode primer + sekunder) g ∈ gains_full — HANYA dijalankan
      kalau gate B0 = GO, kecuali --force-b7b (hasil tetap ditandai "gate B0 ≠ GO").
Metode: GWi_odps, GWi, GWC_tuned, Canny_tuned, Canny_matched (+ Canny_hyst σ-grid untuk
        oracle per level). B7b menambah pipeline pembanding per mode:
        iso → BM3D_Canny (denoise → Canny), dark → ZeroDCE_Canny (enhance → Canny),
        keduanya memakai Canny yang sama dengan Canny_tuned.
Aturan keputusan (config.gates): dievaluasi otomatis → results/b7/decision.json
"""
from __future__ import annotations

import json

import pandas as pd

from _cli import parser, setup, samples_for
from kgexp.common import results_dir, write_json
from kgexp.evaluate import run_counts
from kgexp.methods import tuned_specs
from kgexp.report import plot_levels, save_table, score_table
from kgexp.stats import compare, f_at, score


def oracle_specs(cfg):
    return [{"name": f"Canny_hyst_s{s:g}", "det": "canny_hyst", "pre": cfg["preproc_default"], "sigma": s,
             "hyst_low_ratio": cfg["tuning"]["canny_low_ratio"]} for s in cfg["tuning"]["canny_sigma"]]


def run_levels(cfg, conds, specs, samples, force):
    res = {c: run_counts(cfg, "BSDS500", "test", c, specs, samples, force=force) for c in conds}
    on = [s["name"] for s in specs if s["name"].startswith("Canny_hyst_s")]
    for c in res:
        best = max(on, key=lambda n: score(res[c][n].counts, res[c][n].thresholds).ods)
        res[c]["Canny_oracle"] = res[c][best]
    return res


def tables(cfg, res, level_of, comparators):
    sc, dl, tr = [], [], []
    for c, cs in res.items():
        lv = level_of[c]
        sc.append(score_table({k: v for k, v in cs.items() if not k.startswith("Canny_hyst_s")}, cfg,
                              {"condition": c, "level": lv}))
        for b in comparators:
            if b in cs:
                dl.append({"condition": c, "level": lv, **compare(cs["GWi_odps"], cs[b], cfg), "b": b})
        ref = res[next(iter(res))]   # kondisi pertama = clean / σ=0 / g=1
        for k, v in cs.items():
            if k.startswith("Canny_hyst_s"):
                continue
            t0 = score(ref[k].counts, ref[k].thresholds).ods_t_idx
            tr.append({"condition": c, "level": lv, "method": k, "f_transfer": f_at(v.counts, t0)})
    return pd.concat(sc, ignore_index=True), pd.DataFrame(dl), pd.DataFrame(tr)


def gate(rows, target, comps, levels):
    r = rows[(rows.a == target) & rows.b.isin(comps) & rows.level.isin(levels)]
    complete = len(r) == len(comps) * len(levels)
    ok = complete and bool(((r.delta > 0) & (r.ci_lo > 0)).all())
    return {"decision": "PASS" if ok else ("FAIL" if complete else "INCOMPLETE"),
            "rule": f"ΔODS({target} − c) > 0 dan CI_lo > 0 untuk c ∈ {comps}, level ∈ {levels}",
            "checked": r[["level", "b", "delta", "ci_lo", "ci_hi"]].to_dict("records")}


def main():
    p = parser(__doc__)
    p.add_argument("--skip-b7", action="store_true")
    p.add_argument("--skip-b7b", action="store_true")
    p.add_argument("--force-b7b", action="store_true")
    p.add_argument("--no-bm3d", action="store_true", help="tanpa BM3D (uji cepat; gate iso jadi INCOMPLETE)")
    args = p.parse_args()
    cfg = setup(args)
    out = results_dir(cfg, "b7")
    smp = samples_for(cfg, "BSDS500", "test", args.limit)
    decision = {}
    base = tuned_specs(cfg) + oracle_specs(cfg)
    comps = ["Canny_tuned", "Canny_matched", "Canny_oracle", "GWC_tuned", "GWi"]

    if not args.skip_b7:
        sig = cfg["degradation"]["gaussian_sigmas"]
        conds = {("clean" if s == 0 else f"gauss_s{s:g}"): float(s) for s in sig}
        res = run_levels(cfg, conds, base, smp, args.force)
        sc, dl, tr = tables(cfg, res, conds, comps)
        save_table(sc, out / "b7_scores")
        save_table(dl, out / "b7_deltas")
        save_table(tr, out / "b7_transfer")
        plot_levels(sc, "level", out / "b7_ods_vs_sigma", "σ noise Gaussian (8-bit)",
                    ["GWi_odps", "GWi", "GWC_tuned", "Canny_tuned", "Canny_matched", "Canny_oracle"],
                    title="B7 — BSDS500 test")
        g = cfg["gates"]["b7_noise"]
        decision["b7"] = gate(dl, g["target"], [g["comparator"]], [float(x) for x in g["sigmas"]])
        decision["b7"]["framing"] = "(b) noise" if decision["b7"]["decision"] == "PASS" else "(a) keluarga Gabor"

    if not args.skip_b7b:
        gpath = cfg["paths"]["results"] / "b0" / "gate.json"
        g0 = json.loads(gpath.read_text())["decision"] if gpath.exists() else "MISSING"
        if g0 != "GO" and not args.force_b7b:
            print(f"[B7b] dilewati: gate B0 = {g0}")
            decision["b7b"] = {"decision": "SKIPPED", "b0_gate": g0}
        else:
            ll = cfg["degradation"]["lowlight"]
            # Pipeline pembanding memakai Canny yang SAMA dengan Canny_tuned (σ dari B2)
            canny = {"det": "canny_hyst", "pre": cfg["preproc_default"],
                     "sigma": cfg["selected"].get("Canny", {}).get("sigma", cfg["methods_default"]["canny_sigma"]),
                     "hyst_low_ratio": cfg["tuning"]["canny_low_ratio"]}
            enh_for = {"iso": [] if args.no_bm3d else [{"name": "BM3D_Canny", "enh": "bm3d", **canny}],
                       "dark": [{"name": "ZeroDCE_Canny", "enh": "zerodce", **canny}]}
            for mode in (ll["mode_primary"], ll["mode_secondary"]):
                conds = {"clean": float("inf"), **{f"ll_{mode}_g{x:g}": float(x) for x in ll["gains_full"]}}
                res = run_levels(cfg, conds, base + enh_for[mode], smp, args.force)
                sc, dl, tr = tables(cfg, res, conds, comps + ["ZeroDCE_Canny", "BM3D_Canny"])
                save_table(sc, out / f"b7b_{mode}_scores")
                save_table(dl, out / f"b7b_{mode}_deltas")
                save_table(tr, out / f"b7b_{mode}_transfer")
                plot_levels(sc[sc.level != float("inf")], "level", out / f"b7b_{mode}_ods_vs_gain",
                            "gain g (log)", ["GWi_odps", "GWC_tuned", "Canny_tuned", "Canny_matched",
                                             "BM3D_Canny" if mode == "iso" else "ZeroDCE_Canny", "Canny_oracle"],
                            title=f"B7b — low-light ({mode}), BSDS500 test", log_x=True)
                gg = cfg["gates"]["b7b_lowlight"]
                decision[f"b7b_{mode}"] = gate(dl, gg["target"], gg["comparators"][mode],
                                               [float(x) for x in gg["gains"]])
                decision[f"b7b_{mode}"]["b0_gate"] = g0
            prim = decision.get(f"b7b_{ll['mode_primary']}", {}).get("decision")
            sec = decision.get(f"b7b_{ll['mode_secondary']}", {}).get("decision")
            decision["b7b_framing"] = (
                "low-light (termasuk tanpa kompensasi gain)" if prim == "PASS" and sec == "PASS" else
                "low-light dengan kompensasi gain (auto-exposure) saja" if prim == "PASS" else
                "tidak ada klaim low-light; laporkan B7b sebagai analisis")
    write_json(out / "decision.json", decision)
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
