"""B3 — CI bootstrap ODS dan uji berpasangan (9 perbandingan, koreksi Holm).

Perbandingan per dataset: GWi vs GWC_tuned, GWi_odps vs GWi, GWi_odps vs Canny_tuned.
ΔODS memakai bootstrap berpasangan (indeks citra sama, B = stats.n_boot);
uji per citra = Wilcoxon signed-rank pada F per citra di threshold ODS masing-masing
metode, effect size = rank-biserial r. p di-adjust Holm lintas 9 uji.
Output: results/b3/ods_ci.csv, results/b3/tests.csv, results/b3/claims_guard.json
"""
from __future__ import annotations

import pandas as pd

from _cli import parser, setup, samples_for
from kgexp.common import results_dir, write_json
from kgexp.evaluate import run_counts
from kgexp.methods import tuned_specs
from kgexp.report import save_table, score_table
from kgexp.stats import compare, holm

PAIRS = [("GWi", "GWC_tuned"), ("GWi_odps", "GWi"), ("GWi_odps", "Canny_tuned")]


def main():
    p = parser(__doc__)
    p.add_argument("--datasets", nargs="+", default=["BSDS500", "BIPED", "UDED"])
    args = p.parse_args()
    cfg = setup(args)
    if not cfg["selected"]:
        print("[WARN] selected_config.json kosong → memakai default naskah. Jalankan B1 dan B2 dulu.")
    out = results_dir(cfg, "b3")
    ci_rows, tests = [], []
    for ds in args.datasets:
        cs = run_counts(cfg, ds, "test", "clean", tuned_specs(cfg), samples_for(cfg, ds, "test", args.limit),
                        force=args.force)
        ci_rows.append(score_table(cs, cfg, {"dataset": ds}))
        for a, b in PAIRS:
            tests.append({"dataset": ds, **compare(cs[a], cs[b], cfg)})
    t = pd.DataFrame(tests)
    t["p_holm"] = holm(t.wilcoxon_p.tolist())
    t["ci_excludes_0"] = (t.ci_lo > 0) | (t.ci_hi < 0)
    t["wording"] = [("meningkat" if d > 0 else "menurun") if ex else "tidak berbeda (CI mencakup 0)"
                    for d, ex in zip(t.delta, t.ci_excludes_0)]
    save_table(pd.concat(ci_rows, ignore_index=True), out / "ods_ci")
    save_table(t, out / "tests")
    write_json(out / "claims_guard.json",
               {f"{r.dataset}:{r.a}-{r.b}": r.wording for r in t.itertuples()})
    print(t[["dataset", "a", "b", "delta", "ci_lo", "ci_hi", "p_holm", "r_rb", "wording"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
