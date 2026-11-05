"""C6 — audit angka naskah terhadap CSV hasil.

Setiap klaim numerik di abstrak/teks/kesimpulan didaftarkan di claims.yaml:
    - id: abs_odps_gain_bsds
      text: "+5.7%"                 # string persis seperti di naskah
      expr: "100*(ods('b3/ods_ci','BSDS500','GWi_odps')/ods('b3/ods_ci','BSDS500','GWi')-1)"
      fmt: "+{:.1f}%"
Script menghitung expr dari CSV, memformat dengan fmt, lalu memeriksa:
  (1) nilai terformat == text, (2) text benar-benar muncul di naskah (.docx/.txt/.tex).
    python scripts/c6_audit.py --claims claims.yaml --manuscript naskah.docx
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

EXP = Path(__file__).resolve().parents[1]


def read_text(p: Path) -> str:
    if p.suffix == ".docx":
        from docx import Document
        d = Document(str(p))
        parts = [x.text for x in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts += [c.text for c in row.cells]
        return "\n".join(parts)
    return p.read_text(encoding="utf-8")


def make_env(results: Path):
    cache = {}

    def tbl(name):
        if name not in cache:
            cache[name] = pd.read_csv(results / f"{name}.csv")
        return cache[name]

    def val(name, col, **where):
        t = tbl(name)
        for k, v in where.items():
            t = t[t[k] == v]
        if len(t) != 1:
            raise ValueError(f"{name}{where}: {len(t)} baris")
        return float(t[col].iloc[0])

    def ods(name, dataset, method):
        return val(name, "ods", dataset=dataset, method=method)

    return {"val": val, "ods": ods, "math": math, "abs": abs, "round": round}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", type=Path, default=EXP / "claims.yaml")
    ap.add_argument("--manuscript", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=EXP / "results")
    a = ap.parse_args()
    claims = yaml.safe_load(a.claims.read_text())
    text = read_text(a.manuscript)
    env = make_env(a.results)
    rows = []
    for c in claims:
        try:
            v = eval(c["expr"], {"__builtins__": {}}, env)  # noqa: S307 — expr dari file milik penulis
            got = c["fmt"].format(v)
            rows.append({"id": c["id"], "text": c["text"], "computed": got, "match": got == c["text"],
                         "in_manuscript": c["text"] in text})
        except Exception as e:
            rows.append({"id": c["id"], "text": c["text"], "computed": f"ERROR: {e}", "match": False,
                         "in_manuscript": c["text"] in text})
    df = pd.DataFrame(rows)
    out = a.results / "c6_audit.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    bad = df[~(df.match & df.in_manuscript)]
    if len(bad):
        print(f"\n[FAIL] {len(bad)} klaim tidak cocok. Lihat {out}")
        sys.exit(1)
    print(f"\n[OK] {len(df)} klaim cocok dengan CSV dan ada di naskah.")


if __name__ == "__main__":
    main()
