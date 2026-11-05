"""B6 — tabel operasi per piksel dengan k dan N final dari B1.

Output: results/b6/ops_per_pixel.csv|md — kolom mac_vs_GWC untuk klaim "50% konvolusi"
(GWi/GWC = 0.5 tepat pada MAC filter; operasi non-MAC dicatat terpisah).
"""
from __future__ import annotations

from _cli import parser, setup
from kgexp.common import results_dir
from kgexp.complexity import table
from kgexp.methods import gabor_from, matched_canny_sigma
from kgexp.report import save_table


def main():
    args = parser(__doc__).parse_args()
    cfg = setup(args)
    g = gabor_from(cfg)
    canny = cfg["selected"].get("Canny", {}).get("sigma", cfg["methods_default"]["canny_sigma"])
    log_s = cfg["selected"].get("LoG", {}).get("sigma", cfg["methods_default"]["log_sigma"])
    ms = matched_canny_sigma(g["ksize"], g["wavelength"], g["gamma"])
    df = table(g["ksize"], g["n_orientations"], canny, log_s, ms)
    save_table(df, results_dir(cfg, "b6") / "ops_per_pixel", decimals=3)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
