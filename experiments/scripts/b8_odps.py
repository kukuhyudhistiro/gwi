"""B8 — profiling, vektorisasi, dan uji kesetaraan ODPS; ukur ulang runtime semua metode.

1. Kesetaraan: max|ODPS_lama − ODPS_cepat| harus 0 pada 10 citra BSDS500 test (gagal → exit 1).
2. Profil per baris (line_profiler, bila terpasang) untuk nms_odps.odps dan gwi_argmax.
3. Runtime: harness kgexp.timing (1 thread, warm-up, median 3 putaran) untuk
   GWC, GWi, GWi+ODPS (lama), GWi+ODPS (cepat), Canny_tuned, Canny (legacy) pada subset
   deterministik timing.n_images per dataset.
Output: results/b8/{equivalence.json, timing_raw.csv, timing_summary.csv, line_profile.txt, decision.json}
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd

from _cli import parser, setup
from kgexp.common import odps_distance, results_dir, write_json
from kgexp.data import list_samples, load_rgb, subset
from kgexp.methods import finish_pre, gabor_from, gwi_argmax, to_gray_u8, tuned_specs
from kgexp.odps_fast import odps_fast
from kgexp.timing import run_timing, summarize

import gabor_core as gc  # noqa: E402
import nms_odps  # noqa: E402


def equivalence(cfg, n=10):
    g = gabor_from(cfg)
    bank = gc.GWiFilterBank(gc.GaborParams(ksize=g["ksize"], wavelength=g["wavelength"]))
    d = odps_distance(g["wavelength"], cfg["methods_default"]["odps_d"])
    worst = 0.0
    for s in list_samples(cfg, "BSDS500", "test")[:n]:
        img = finish_pre(to_gray_u8(load_rgb(s)), cfg["preproc_default"])
        mag, oi = gwi_argmax(img, bank, g.get("border", "constant"))
        a = nms_odps.odps(mag, oi, n_orientations=8, d=d)
        b = odps_fast(mag, oi, n_orientations=8, d=d)
        worst = max(worst, float(np.abs(a - b).max()))
    return {"n_images": n, "max_abs_diff": worst, "pass": worst == 0.0}


def line_profile(cfg):
    try:
        from line_profiler import LineProfiler
    except ImportError:
        return "line_profiler tidak terpasang (pip install line_profiler)"
    g = gabor_from(cfg)
    bank = gc.GWiFilterBank(gc.GaborParams(ksize=g["ksize"], wavelength=g["wavelength"]))
    s = list_samples(cfg, "BSDS500", "test")[0]
    img = finish_pre(to_gray_u8(load_rgb(s)), cfg["preproc_default"])
    lp = LineProfiler(nms_odps.odps, nms_odps.compute_normal_offsets, gwi_argmax, odps_fast)
    for _ in range(20):
        mag, oi = lp(gwi_argmax)(img, bank, g.get("border", "constant"))
        lp(nms_odps.odps)(mag, oi)
        lp(odps_fast)(mag, oi)
    buf = io.StringIO()
    lp.print_stats(stream=buf)
    return buf.getvalue()


def main():
    p = parser(__doc__)
    p.add_argument("--datasets", nargs="+", default=["BSDS500", "BIPED", "UDED"])
    args = p.parse_args()
    cfg = setup(args)
    out = results_dir(cfg, "b8")
    eq = equivalence(cfg)
    write_json(out / "equivalence.json", eq)
    print("[B8] kesetaraan:", eq)
    if not eq["pass"]:
        sys.exit(1)
    (out / "line_profile.txt").write_text(line_profile(cfg))

    t = tuned_specs(cfg)
    by = {s["name"]: s for s in t}
    specs = [by["GWC_tuned"], by["GWi"], by["GWi_odps"],
             {**by["GWi_odps"], "name": "GWi_odps_fast", "odps_impl": "fast"},
             by["Canny_tuned"],
             {"name": "Canny", "det": "canny_legacy", "pre": cfg["preproc_default"],
              "sigma": cfg["methods_default"]["canny_sigma"]}]
    raws = []
    for ds in args.datasets:
        try:
            smp = subset(list_samples(cfg, ds, "test", args.limit), cfg["timing"]["n_images"], cfg["seed"])
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[WARN] {ds}: {e}")
            continue
        df = run_timing(cfg, smp, specs)
        df["dataset"] = ds
        raws.append(df)
    raw = pd.concat(raws, ignore_index=True)
    raw.to_csv(out / "timing_raw.csv", index=False)
    summ = pd.concat([summarize(r).assign(dataset=r.dataset.iloc[0]) for r in raws], ignore_index=True)
    summ.to_csv(out / "timing_summary.csv", index=False)
    print(summ[["dataset", "method", "total_s_ms_median", "det_s_ms_median", "odps_s_ms_median"]].round(2).to_string(index=False))

    dec = {}
    for ds, grp in summ.groupby("dataset"):
        m = grp.set_index("method")["total_s_ms_median"]
        fast = m.get("GWi_odps_fast")
        dec[ds] = {"GWi_odps_fast_ms": fast, "GWC_ms": m.get("GWC_tuned"),
                   "abstract_wording": "faster than complex Gabor" if fast < m.get("GWC_tuned") else "comparable"}
    write_json(out / "decision.json", dec)
    print(dec)


if __name__ == "__main__":
    main()
