"""Dataset sintetis berstruktur BSDS500/BIPED/UDED untuk uji asap (smoke test) pipeline.
BUKAN untuk hasil. Citra = poligon/elips bertekstur, GT = boundary dari peta label,
BSDS diberi 2 anotator (salah satunya digeser 1 piksel).

    python tools/make_synthetic_data.py --out /tmp/synth_data --n 6
    python scripts/m9_run_all.py --extra --config tests/config_synth.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat
from skimage.segmentation import find_boundaries


def scene(rng, h=160, w=240):
    lab = np.zeros((h, w), np.int32)
    yy, xx = np.mgrid[:h, :w]
    for k in range(1, rng.integers(4, 8)):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        ry, rx = rng.uniform(15, 60), rng.uniform(15, 80)
        a = rng.uniform(0, np.pi)
        u = ((xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)) / rx
        v = (-(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)) / ry
        lab[u ** 2 + v ** 2 < 1] = k
    cols = rng.uniform(30, 225, size=(lab.max() + 1, 3))
    img = cols[lab] + rng.normal(0, 6, (h, w, 3)) + 10 * np.sin(xx / 7.0)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8), lab


def bsds_mat(path, labs):
    gt = np.empty((1, len(labs)), dtype=object)
    for i, lab in enumerate(labs):
        b = find_boundaries(lab, mode="inner").astype(np.uint8)
        gt[0, i] = {"Segmentation": lab.astype(np.uint16), "Boundaries": b}
    savemat(str(path), {"groundTruth": gt})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for d in ("images", "groundTruth"):
            (a.out / "BSDS500" / d / split).mkdir(parents=True, exist_ok=True)
        for i in range(a.n):
            img, lab = scene(rng)
            Image.fromarray(img).save(a.out / "BSDS500" / "images" / split / f"{split}{i:03d}.jpg", quality=95)
            bsds_mat(a.out / "BSDS500" / "groundTruth" / split / f"{split}{i:03d}.mat", [lab, np.roll(lab, 1, 1)])
    for ds, sub in [("BIPED", {"train": ("edges/imgs/train/rgbr/real", "edges/edge_maps/train/rgbr/real"),
                               "test": ("edges/imgs/test", "edges/edge_maps/test")}),
                    ("UDED", {"test": ("imgs", "gt")})]:
        for split, (im, gt) in sub.items():
            (a.out / ds / im).mkdir(parents=True, exist_ok=True)
            (a.out / ds / gt).mkdir(parents=True, exist_ok=True)
            for i in range(a.n):
                img, lab = scene(rng)
                Image.fromarray(img).save(a.out / ds / im / f"{ds}{split}{i:03d}.jpg", quality=95)
                b = (find_boundaries(lab, mode="inner") * 255).astype(np.uint8)
                Image.fromarray(b).save(a.out / ds / gt / f"{ds}{split}{i:03d}.png")
    print(f"[OK] {a.out}")


if __name__ == "__main__":
    main()
