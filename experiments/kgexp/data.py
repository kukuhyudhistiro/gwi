"""Daftar sampel per dataset/split dan loader citra + GT."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from .common import legacy_eval_module

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif")


@dataclass(frozen=True)
class Sample:
    dataset: str
    split: str
    image_id: str
    image_path: Path
    gt_path: Path
    gt_format: str


def list_samples(cfg, dataset: str, split: str, limit: int | None = None) -> List[Sample]:
    dcfg = cfg["datasets"][dataset]
    if split not in dcfg["splits"]:
        raise KeyError(f"{dataset} tidak punya split '{split}'")
    sp = dcfg["splits"][split]
    root = cfg["paths"]["data_root"] / dataset
    img_dir, gt_dir = root / sp["images"], root / sp["gt"]
    if not gt_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"{dataset}/{split}: {img_dir} atau {gt_dir} tidak ada")
    ext = ".mat" if dcfg["gt_format"] == "mat" else ".png"
    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    out = []
    for g in sorted(gt_dir.glob(f"*{ext}")):
        if g.stem in images:
            out.append(Sample(dataset, split, g.stem, images[g.stem], g, dcfg["gt_format"]))
    if not out:
        raise RuntimeError(f"{dataset}/{split}: tidak ada pasangan citra–GT")
    return out[:limit] if limit else out


def subset(samples: List[Sample], n: int, seed: int) -> List[Sample]:
    """Subset deterministik (urutan id) — untuk pilot/timing."""
    if n >= len(samples):
        return samples
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(samples), size=n, replace=False))
    return [samples[i] for i in idx]


def load_rgb(sample: Sample) -> np.ndarray:
    return np.array(Image.open(str(sample.image_path)).convert("RGB"))


def load_gts(sample: Sample) -> List[np.ndarray]:
    ev = legacy_eval_module()
    if sample.gt_format == "mat":
        return ev.load_gt_bsds_mat(sample.gt_path)
    return ev.load_gt_png(sample.gt_path)
