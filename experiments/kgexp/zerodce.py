"""Zero-DCE (Guo et al., CVPR 2020) — inferensi NumPy tanpa PyTorch.

Arsitektur = enhance_net_nopool pada repo resmi Li-Chongyi/Zero-DCE (7 conv 3×3, 32 kanal,
8 iterasi kurva LE(x) = x + r·(x² − x)). Bobot resmi (snapshots/Epoch99.pth) dikonversi
sekali ke .npz dengan tools/convert_zerodce.py.

Catatan runtime: implementasi NumPy ini dipakai untuk AKURASI. Untuk latency di Tabel
runtime, ukur juga versi PyTorch resmi (single thread) kalau tersedia, dan laporkan
keduanya dengan label yang jelas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ZeroDCE:
    def __init__(self, weights_npz: Path):
        if not Path(weights_npz).exists():
            raise FileNotFoundError(
                f"{weights_npz} tidak ada. Jalankan: python tools/convert_zerodce.py "
                "--pth <Zero-DCE>/Zero-DCE_code/snapshots/Epoch99.pth")
        w = np.load(weights_npz)
        self.w = {k: w[k].astype(np.float32) for k in w.files}

    @staticmethod
    def _conv3x3(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        # x: (C, H, W) ; W: (Cout, C, 3, 3) ; padding 1, stride 1
        c, h, wd = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        out = np.empty((W.shape[0], h * wd), dtype=np.float32)
        out[:] = b[:, None]
        for dy in range(3):
            for dx in range(3):
                patch = xp[:, dy:dy + h, dx:dx + wd].reshape(c, -1)
                out += W[:, :, dy, dx] @ patch
        return out.reshape(-1, h, wd)

    def _layer(self, name, x, act="relu"):
        y = self._conv3x3(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"])
        return np.maximum(y, 0, out=y) if act == "relu" else np.tanh(y)

    def enhance(self, rgb_u8: np.ndarray) -> np.ndarray:
        x = (rgb_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)
        x1 = self._layer("e_conv1", x)
        x2 = self._layer("e_conv2", x1)
        x3 = self._layer("e_conv3", x2)
        x4 = self._layer("e_conv4", x3)
        x5 = self._layer("e_conv5", np.concatenate([x3, x4], 0))
        x6 = self._layer("e_conv6", np.concatenate([x2, x5], 0))
        xr = self._layer("e_conv7", np.concatenate([x1, x6], 0), act="tanh")
        y = x
        for i in range(8):
            r = xr[3 * i:3 * i + 3]
            y = y + r * (y * y - y)
        y = np.clip(y, 0.0, 1.0).transpose(1, 2, 0)
        return np.round(y * 255.0).astype(np.uint8)
