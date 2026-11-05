"""Konfigurasi, path, seed deterministik, dan impor modul legacy repo."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

EXP_DIR = Path(__file__).resolve().parents[1]      # <repo>/experiments
REPO_DIR = EXP_DIR.parent                          # <repo>
# Modul legacy (gabor_core, nms_odps, baselines) dan evaluator 04c. Dua layout didukung:
#   repo GitHub gwi-odps : <repo>/src/*.py + <repo>/scripts/04c_evaluate_bsds_v2.py
#   folder kerja lokal   : <repo>/tools/scripts/ (semuanya di satu folder)
# Kode metodenya identik di kedua layout (beda hanya docstring, dicek 2026-11-05).
if (REPO_DIR / "src" / "gabor_core.py").exists():
    SRC_DIR, LEGACY_SCRIPTS_DIR = REPO_DIR / "src", REPO_DIR / "scripts"
else:
    SRC_DIR = LEGACY_SCRIPTS_DIR = REPO_DIR / "tools" / "scripts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    path = Path(path) if path else EXP_DIR / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = (p if p.is_absolute() else (EXP_DIR / p)).resolve()
    sel = cfg["paths"]["results"] / "selected_config.json"
    cfg["selected"] = json.loads(sel.read_text()) if sel.exists() else {}
    return cfg


def results_dir(cfg, *parts) -> Path:
    p = cfg["paths"]["results"].joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(obj: Any, n: int = 10) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def image_rng(seed: int, image_id: str, condition: str) -> np.random.Generator:
    """RNG deterministik per (seed, citra, kondisi) — tidak bergantung urutan proses."""
    return np.random.default_rng([seed, zlib.crc32(image_id.encode()),
                                  zlib.crc32(condition.encode())])


def odps_distance(wavelength: float, setting="auto") -> int:
    if setting != "auto":
        return int(setting)
    return int(np.floor(wavelength / 2.0 + 0.5))   # λ=4 → 2 (paritas naskah)


_LEGACY_EVAL = None


def legacy_eval_module():
    """Impor scripts/04c_evaluate_bsds_v2.py (nama file diawali angka)."""
    global _LEGACY_EVAL
    if _LEGACY_EVAL is None:
        f = LEGACY_SCRIPTS_DIR / "04c_evaluate_bsds_v2.py"
        spec = importlib.util.spec_from_file_location("eval04c", f)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["eval04c"] = mod          # dataclass butuh modul terdaftar
        spec.loader.exec_module(mod)
        _LEGACY_EVAL = mod
    return _LEGACY_EVAL


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_np_default))


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
