"""B0 — pilot low-light (≈2 HK) dengan aturan go/no-go yang sudah ditetapkan di config.yaml.

Data   : BSDS500 **val** (100 citra) — test set tidak disentuh sebelum desain final.
Kondisi: clean + low-light mode iso dan dark, g ∈ gains_pilot (config).
Metode : GWi_odps, GWi, GWC (konfigurasi Paper 2, methods_default), Canny NMS σ ∈ grid (σ dipilih di CLEAN →
         Canny_clean_selected; σ terbaik per kondisi → Canny_oracle, kontrol saja),
         Canny_matched (σ disamakan dengan support kernel GWi),
         BM3D_Canny  (denoise → Canny, σ = Canny_clean_selected)   ← pembanding mode iso
         ZeroDCE_Canny (enhance → Canny, σ = Canny_clean_selected) ← pembanding mode dark
Ukuran : ODS + CI bootstrap, ΔODS berpasangan, dan F pada threshold transfer
         (threshold ODS dari kondisi clean dipakai tanpa diubah).
Gate   : config.gates.b0_pilot (keputusan dari mode `decide_on`; mode lain = sekunder).
Output : results/b0/{scores,deltas,transfer}.csv|md, gate.json, plot_ods_vs_gain_<mode>.png

    python scripts/b0_pilot.py                    # pre = preproc_default (he_cv2, Paper 2) dan none
    python scripts/b0_pilot.py --preproc he_cv2   # protokol utama saja (lebih cepat)
    python scripts/b0_pilot.py --no-bm3d          # tanpa BM3D (uji cepat; gate iso jadi INCOMPLETE)
"""
from __future__ import annotations

import json

import pandas as pd

from _cli import parser, setup, samples_for
from kgexp.common import results_dir, write_json
from kgexp.evaluate import run_counts
from kgexp.report import plot_levels, save_table, score_table
from kgexp.stats import compare, f_at, score

INF = float("inf")


def base_specs(cfg, pre):
    md, tu = cfg["methods_default"], cfg["tuning"]
    g = md["gabor"]
    sp = [
        {"name": "GWi_odps", "det": "gwi_odps", "pre": pre, "odps_d": md["odps_d"], **g},
        {"name": "GWi", "det": "gwi", "pre": pre, **g},
        {"name": "GWC", "det": "gwc", "pre": pre, **g},
        {"name": "Canny", "det": "canny_legacy", "pre": pre, "sigma": md["canny_sigma"]},
        {"name": "Canny_matched", "det": "canny_nms", "pre": pre, "sigma": "matched",
         "ksize": g["ksize"], "wavelength": g["wavelength"]},
    ]
    sp += [{"name": f"Canny_nms_s{s:g}", "det": "canny_nms", "pre": pre, "sigma": s} for s in tu["canny_sigma"]]
    return sp


def enh_specs(pre, sigma, mode, with_bm3d):
    sp = []
    if mode == "iso" and with_bm3d:
        sp.append({"name": "BM3D_Canny", "det": "canny_nms", "pre": pre, "enh": "bm3d", "sigma": sigma})
    if mode == "dark":
        sp.append({"name": "ZeroDCE_Canny", "det": "canny_nms", "pre": pre, "enh": "zerodce", "sigma": sigma})
    return sp


def evaluate_gate(gcfg, deltas: pd.DataFrame, mode: str):
    comps = gcfg["comparators"][mode]
    rows = deltas[(deltas["mode"] == mode) & (deltas.a == gcfg["target"]) & deltas.b.isin(comps)
                  & deltas.gain.isin(gcfg["gains"])]
    need = len(comps) * len(gcfg["gains"])
    complete = len(rows) == need
    go = complete and bool((rows.delta > 0).all())
    return {"decision": ("GO" if go else "NO-GO") if complete else "INCOMPLETE",
            "rule": f"[{mode}] ΔODS({gcfg['target']} − c) > 0 untuk semua c ∈ {comps}, g ∈ {gcfg['gains']}",
            "checked": rows[["gain", "b", "delta", "ci_lo", "ci_hi"]].to_dict("records"),
            "missing": need - len(rows)}


def main():
    p = parser(__doc__)
    p.add_argument("--preproc", nargs="+", default=None,
                   help="bawaan: preproc_default (config) dan none")
    p.add_argument("--no-bm3d", action="store_true")
    p.add_argument("--split", default="val")
    args = p.parse_args()
    cfg = setup(args)
    if args.preproc is None:
        args.preproc = list(dict.fromkeys([cfg["preproc_default"], "none"]))
    ll = cfg["degradation"]["lowlight"]
    gains = ll["gains_pilot"]
    modes = [ll["mode_primary"], ll["mode_secondary"]]
    samples = samples_for(cfg, "BSDS500", args.split, args.limit)
    out = results_dir(cfg, "b0")
    S, D, T, per_pre = [], [], [], {}

    for pre in args.preproc:
        specs = base_specs(cfg, pre)
        cn = [s["name"] for s in specs if s["name"].startswith("Canny_nms_s")]
        # Pass 1 — clean: pilih σ Canny (bukan oracle per kondisi)
        clean = run_counts(cfg, "BSDS500", args.split, "clean", specs, samples, force=args.force)
        best = max(cn, key=lambda n: score(clean[n].counts, clean[n].thresholds).ods)
        sigma = next(s["sigma"] for s in specs if s["name"] == best)
        per_pre[pre] = {"canny_clean_selected": best, "sigma": sigma}

        for mode in modes:
            esp = enh_specs(pre, sigma, mode, not args.no_bm3d)
            conds = {"clean": INF, **{f"ll_{mode}_g{g:g}": float(g) for g in gains}}
            res = {c: run_counts(cfg, "BSDS500", args.split, c, specs + esp, samples, force=args.force)
                   for c in conds}
            for c in res:
                res[c]["Canny_clean_selected"] = res[c][best]
                orc = max(cn, key=lambda n: score(res[c][n].counts, res[c][n].thresholds).ods)
                res[c]["Canny_oracle"] = res[c][orc]
            for c, g in conds.items():
                tag = {"pre": pre, "mode": mode, "condition": c, "gain": g}
                S.append(score_table({k: v for k, v in res[c].items() if not k.startswith("Canny_nms_s")},
                                     cfg, tag))
                for b in ["Canny_clean_selected", "Canny_matched", "BM3D_Canny", "ZeroDCE_Canny",
                          "Canny_oracle", "GWC", "GWi"]:
                    if b in res[c]:
                        D.append({**tag, **compare(res[c]["GWi_odps"], res[c][b], cfg), "b": b})
                for name, cs in res[c].items():
                    if name.startswith("Canny_nms_s"):
                        continue
                    t0 = score(res["clean"][name].counts, res["clean"][name].thresholds).ods_t_idx
                    T.append({**tag, "method": name, "f_transfer": f_at(cs.counts, t0),
                              "ods": score(cs.counts, cs.thresholds).ods})

    scores, deltas, transfer = pd.concat(S, ignore_index=True), pd.DataFrame(D), pd.DataFrame(T)
    save_table(scores, out / "scores")
    save_table(deltas, out / "deltas")
    save_table(transfer, out / "transfer")

    gcfg = cfg["gates"]["b0_pilot"]
    dmain = deltas[deltas.pre == gcfg["preproc"]]
    per_mode = {m: evaluate_gate(gcfg, dmain, m) for m in modes}
    gate = {"decision": per_mode[gcfg["decide_on"]]["decision"], "decide_on": gcfg["decide_on"],
            "per_mode": per_mode, "split": args.split, "n_images": len(samples), "limit": args.limit,
            "per_preproc": per_pre,
            "note": "Pilot memutuskan arah, bukan klaim. Klaim final hanya dari B7b pada test set."}
    if "none" in args.preproc and gcfg["preproc"] != "none":
        dn = deltas[deltas.pre == "none"]
        gate["sensitivity_no_he"] = {m: evaluate_gate(gcfg, dn, m)["decision"] for m in modes}
    write_json(out / "gate.json", gate)

    for mode in modes:
        pl = scores[(scores.pre == gcfg["preproc"]) & (scores["mode"] == mode) & (scores.gain != INF)]
        plot_levels(pl, "gain", out / f"plot_ods_vs_gain_{mode}", "gain g (log; kiri = lebih gelap)",
                    ["GWi_odps", "GWC", "Canny_clean_selected", "Canny_matched",
                     "BM3D_Canny" if mode == "iso" else "ZeroDCE_Canny", "Canny_oracle"],
                    title=f"B0 pilot — BSDS500 {args.split}, {mode}, pre={gcfg['preproc']}", log_x=True)
    print(json.dumps({k: v for k, v in gate.items() if k != "per_mode"} |
                     {"per_mode": {m: v["decision"] for m, v in per_mode.items()}}, indent=2, default=str))


if __name__ == "__main__":
    main()
