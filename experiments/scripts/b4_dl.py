"""B4 — detektor DL ringan dengan protokol evaluasi yang sama.

Langkah di luar script ini (HK6–HK8): clone repo resmi tiap metode, unduh bobot pretrained
RESMI, jalankan inferensi di CPU (torch.set_num_threads(1)), simpan peta probabilitas
8-bit ke results/maps/<metode>/<dataset>/<image_id>.png (resolusi asli, tanpa NMS khusus).
Catat dataset training tiap bobot di config (b4.methods) — hasil BIPED untuk bobot yang
dilatih di BIPED train diberi tanda.

Script ini:
  (1) --eval     : evaluasi peta dengan evaluator yang sama (thinning + matching), simpan cache count
  (2) --profile  : parameter, FLOPs (thop/fvcore bila ada), ms/citra (median 3 putaran setelah
                   3 warm-up, 1 thread), peak RSS. Butuh PyTorch + modul adapter:
                   --model "paket.modul:fungsi_pembuat_model" --weights path.pth
  (3) --pareto   : plot ODS vs ms/citra (log) untuk semua metode yang punya runtime

    python scripts/b4_dl.py --eval --methods TEED PiDiNet_tiny LDC
    python scripts/b4_dl.py --profile --name TEED --model teed_adapter:build --weights teed.pth
"""
from __future__ import annotations

import importlib
import json
import time

import numpy as np
import pandas as pd

from _cli import parser, setup
from kgexp.common import results_dir
from kgexp.data import list_samples, load_rgb, subset
from kgexp.evaluate import counts_from_maps
from kgexp.report import save_table, score_table


def do_eval(cfg, methods, datasets):
    root = cfg["b4"]["maps_root"] if "maps_root" in cfg["b4"] else cfg["paths"]["results"] / "maps"
    rows = []
    for m in methods:
        for ds in datasets:
            d = (cfg["paths"]["results"] / "maps" / m / ds)
            if not d.exists():
                print(f"[WARN] {d} tidak ada")
                continue
            cs = counts_from_maps(cfg, ds, "test", d, m)
            rows.append(score_table({m: cs}, cfg, {"dataset": ds}))
    if rows:
        df = pd.concat(rows, ignore_index=True)
        save_table(df, results_dir(cfg, "b4") / "dl_scores")
        print(df.round(4).to_string(index=False))


def do_profile(cfg, name, model_ref, weights, dataset):
    import psutil
    import torch
    torch.set_num_threads(cfg["b4"]["threads"])
    mod, fn = model_ref.split(":")
    model = getattr(importlib.import_module(mod), fn)(weights)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    samples = subset(list_samples(cfg, dataset, "test"), cfg["timing"]["n_images"], cfg["seed"])
    x0 = load_rgb(samples[0])
    inp = torch.from_numpy(x0.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    flops = None
    try:
        from thop import profile as thop_profile
        flops = float(thop_profile(model, inputs=(inp,), verbose=False)[0]) * 2   # MAC → FLOPs
    except Exception as e:  # thop opsional
        print(f"[INFO] FLOPs tidak dihitung ({e})")
    proc = psutil.Process()
    times, rss = [], []
    with torch.inference_mode():
        for s in samples[: cfg["b4"]["warmup"]]:
            x = load_rgb(s)
            model(torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0))
        for s in samples:
            x = load_rgb(s)
            t_in = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
            rep = []
            for _ in range(cfg["b4"]["repeats"]):
                t0 = time.perf_counter()
                model(t_in)
                rep.append(time.perf_counter() - t0)
            times.append(np.median(rep) * 1e3)
            rss.append(proc.memory_info().rss / 2 ** 20)
    res = {"method": name, "dataset": dataset, "params": n_params, "flops_first_image": flops,
           "ms_median": float(np.median(times)), "ms_mean": float(np.mean(times)),
           "ms_sd": float(np.std(times)), "peak_rss_mb": float(max(rss)), "threads": cfg["b4"]["threads"],
           "note": "FLOPs pada resolusi citra pertama; preprocessing/postprocessing resmi tidak termasuk"}
    out = results_dir(cfg, "b4") / f"profile_{name}_{dataset}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def do_pareto(cfg):
    import glob
    from kgexp.report import PALETTE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b4 = results_dir(cfg, "b4")
    rt = cfg["paths"]["results"] / "b8" / "timing_summary.csv"
    sc = cfg["paths"]["results"] / "b3" / "ods_ci.csv"
    pts = []
    if rt.exists() and sc.exists():
        r, s = pd.read_csv(rt), pd.read_csv(sc)
        s = s[s.dataset == "BSDS500"]
        for _, row in s.iterrows():
            m = r[r.method == row.method]
            if len(m):
                pts.append((row.method, float(m.total_s_ms_median.iloc[0]), row.ods, "klasik"))
    dl = b4 / "dl_scores.csv"
    dls = pd.read_csv(dl) if dl.exists() else pd.DataFrame()
    for f in glob.glob(str(b4 / "profile_*_BSDS500.json")):
        p = json.loads(open(f).read())
        o = dls[(dls.method == p["method"]) & (dls.dataset == "BSDS500")]
        if len(o):
            pts.append((p["method"], p["ms_median"], float(o.ods.iloc[0]), "DL"))
    if not pts:
        print("[WARN] belum ada titik untuk Pareto (butuh b3/ods_ci.csv, b8/timing_summary.csv, profil DL)")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    for name, ms, ods, kind in pts:
        ax.scatter(ms, ods, s=36, color=PALETTE[0] if kind == "klasik" else PALETTE[1],
                   marker="o" if kind == "klasik" else "s", zorder=3)
        ax.annotate(name, (ms, ods), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("ms / citra (median, CPU 1 thread, log)")
    ax.set_ylabel("ODS BSDS500 test")
    ax.grid(alpha=0.25, lw=0.6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(b4 / "pareto.png")
    fig.savefig(b4 / "pareto.pdf")
    pd.DataFrame(pts, columns=["method", "ms", "ods", "kind"]).to_csv(b4 / "pareto_points.csv", index=False)


def main():
    p = parser(__doc__)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--pareto", action="store_true")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=["BSDS500", "BIPED", "UDED"])
    p.add_argument("--name")
    p.add_argument("--model")
    p.add_argument("--weights")
    p.add_argument("--profile-dataset", default="BSDS500")
    args = p.parse_args()
    cfg = setup(args)
    if args.eval:
        do_eval(cfg, args.methods or cfg["b4"]["methods"], args.datasets)
    if args.profile:
        do_profile(cfg, args.name, args.model, args.weights, args.profile_dataset)
    if args.pareto:
        do_pareto(cfg)


if __name__ == "__main__":
    main()
