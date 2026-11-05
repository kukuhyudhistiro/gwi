"""M0c — reproduksi tahap GWi Paper 2 dengan pipeline di repo ini.

Tahap GWi paket ini diambil dari Paper 2:
    Yudhistiro, Basuki, Rijati (2026). Adaptive Imaginary Gabor Wavelet. JESA 59(9):2495–2508.
    doi:10.18280/jesa.590906. Kode: github.com/kukuhyudhistiro/localgradientactivity-
Skrip ini mengecek bahwa GWi dan GWC (k = 7, λ = 4, σ = 0.56·λ, γ = 0.5, N = 8, reflection
padding, HE OpenCV) menghasilkan ODS yang sama dengan Tabel 5–6 Paper 2. Sobel dan Canny
(magnitudo gradien, σ = 1.4) ikut dicek sebagai kontrol preprocessing.

Acuan: results/tables/main_results_ods_summary.csv di repo Paper 2 (presisi penuh).
Kriteria lulus: |ΔODS| ≤ --tol untuk GWi dan GWC di kedua dataset. Sobel dan Canny dilaporkan
saja, karena definisi Canny di naskah dan di kode Paper 2 tidak sepenuhnya jelas.

    python scripts/m0_paper2_regression.py
"""
from __future__ import annotations

import sys

import pandas as pd

from _cli import parser, setup, samples_for
from kgexp.common import results_dir, write_json
from kgexp.evaluate import run_counts
from kgexp.report import save_table, score_table

# ODS Paper 2 (main_results_ods_summary.csv, kolom ods_f)
PAPER2_ODS = {
    "BSDS500": {"GWi": 0.417139121, "GWC": 0.396538338, "Sobel": 0.463537635, "Canny": 0.532914546},
    "UDED":    {"GWi": 0.579875858, "GWC": 0.506451829, "Sobel": 0.637437656, "Canny": 0.670059808},
}
GATED = ("GWi", "GWC")


def paper2_specs(cfg):
    g, pre = cfg["methods_default"]["gabor"], cfg["preproc_default"]
    return [
        {"name": "GWi", "det": "gwi", "pre": pre, **g},
        {"name": "GWC", "det": "gwc", "pre": pre, **g},
        {"name": "Sobel", "det": "sobel", "pre": pre, "sigma": 0.0},
        {"name": "Canny", "det": "canny_legacy", "pre": pre, "sigma": 1.4},
    ]


def main():
    p = parser(__doc__)
    p.add_argument("--datasets", nargs="+", default=["BSDS500", "UDED"])
    p.add_argument("--tol", type=float, default=2e-3)
    args = p.parse_args()
    cfg = setup(args)
    g = cfg["methods_default"]["gabor"]
    if (g["ksize"], g["wavelength"], g.get("border")) != (7, 4.0, "reflect") or cfg["preproc_default"] != "he_cv2":
        print(f"[WARN] config bukan konfigurasi Paper 2: gabor={g}, pre={cfg['preproc_default']}")
    rows = []
    for ds in args.datasets:
        cs = run_counts(cfg, ds, "test", "clean", paper2_specs(cfg), samples_for(cfg, ds, "test", args.limit),
                        force=args.force)
        df = score_table(cs, cfg, {"dataset": ds})
        df["ods_paper2"] = df.method.map(PAPER2_ODS[ds])
        df["delta"] = df.ods - df.ods_paper2
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["gated"] = df.method.isin(GATED)
    df["pass"] = ~df.gated | (df.delta.abs() <= args.tol)
    out = results_dir(cfg, "m0")
    save_table(df[["dataset", "method", "n", "ods", "ods_paper2", "delta", "ois", "ap", "gated", "pass"]],
               out / "paper2_regression", decimals=4)
    ok = bool(df["pass"].all()) and not args.limit
    write_json(out / "paper2_regression.json",
               {"pass": ok, "tol": args.tol, "limit": args.limit,
                "max_abs_delta_gated": float(df[df.gated].delta.abs().max())})
    print(df[["dataset", "method", "ods", "ods_paper2", "delta", "pass"]].to_string(index=False))
    print("[M0c]", "LULUS" if ok else "GAGAL", f"(tol {args.tol})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
