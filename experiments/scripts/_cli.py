"""Argumen bersama untuk semua script modul."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kgexp.common import load_config  # noqa: E402


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--n-jobs", type=int, default=None, help="override eval.n_jobs")
    p.add_argument("--limit", type=int, default=None,
                   help="batasi jumlah citra per split (HANYA untuk uji cepat; jangan dilaporkan)")
    p.add_argument("--force", action="store_true", help="abaikan cache")
    return p


def setup(args):
    cfg = load_config(args.config)
    if args.n_jobs:
        cfg["eval"]["n_jobs"] = args.n_jobs
    if args.limit:
        print(f"[WARN] --limit {args.limit}: hasil hanya untuk uji cepat, bukan untuk naskah.")
    return cfg


def samples_for(cfg, dataset, split, limit):
    from kgexp.data import list_samples
    return list_samples(cfg, dataset, split, limit=limit)
