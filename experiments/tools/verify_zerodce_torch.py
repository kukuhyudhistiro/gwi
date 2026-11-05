"""Verifikasi implementasi NumPy Zero-DCE (kgexp/zerodce.py) terhadap model PyTorch resmi.

Butuh PyTorch dan clone repo resmi:
    git clone --depth 1 https://github.com/Li-Chongyi/Zero-DCE
    python tools/verify_zerodce_torch.py --repo Zero-DCE --n 5
Kriteria lolos: selisih maksimum ≤ 1 level abu-abu (pembulatan 8-bit) pada semua citra,
untuk citra bersih dan citra low-light mode dark (g = 0.05).
Hasil pengujian 2026-11-04 (BSDS500 val, 3 citra × 2 kondisi): selisih maksimum = 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from kgexp.common import load_config  # noqa: E402
from kgexp.data import list_samples, load_rgb, subset  # noqa: E402
from kgexp.degrade import simulate_lowlight  # noqa: E402
from kgexp.zerodce import ZeroDCE  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True, help="folder clone Zero-DCE")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    import torch
    sys.path.insert(0, str(a.repo / "Zero-DCE_code"))
    import model as zm
    torch.set_num_threads(1)
    net = zm.enhance_net_nopool()
    net.load_state_dict(torch.load(a.repo / "Zero-DCE_code" / "snapshots" / "Epoch99.pth",
                                   map_location="cpu"))
    net.eval()
    cfg = load_config(a.config)
    npz = ZeroDCE(cfg["paths"]["zerodce_weights"])
    worst = 0
    for s in subset(list_samples(cfg, "BSDS500", "val"), a.n, cfg["seed"]):
        rgb = load_rgb(s)
        dark = simulate_lowlight(rgb, 0.05, np.random.default_rng(0), mode="dark")
        for name, im in (("clean", rgb), ("dark_g0.05", dark)):
            x = torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
            with torch.no_grad():
                out = net(x)
            y = out[1] if isinstance(out, (tuple, list)) else out
            yt = np.round(np.clip(y[0].permute(1, 2, 0).numpy(), 0, 1) * 255).astype(int)
            d = int(np.abs(yt - npz.enhance(im).astype(int)).max())
            worst = max(worst, d)
            print(f"{s.image_id} {name:11s} max|diff| = {d}")
    print(f"[{'OK' if worst <= 1 else 'FAIL'}] selisih maksimum = {worst}")
    sys.exit(0 if worst <= 1 else 1)


if __name__ == "__main__":
    main()
