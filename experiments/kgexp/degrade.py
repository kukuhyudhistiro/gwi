"""Model degradasi citra: noise Gaussian aditif (B7) dan low-light Poisson–Gaussian (B0/B7b).

Semua fungsi menerima RGB uint8 dan mengembalikan RGB uint8, sehingga preprocessing
naskah (BT.601 + HE) tetap dijalankan sesudahnya tanpa perubahan.

Model low-light (lihat config.yaml → degradation.lowlight):
    x_lin = (x/255)^γ
    e     = Poisson(g · FW · x_lin) + N(0, σ_r²)              [elektron]
    iso : y_lin = e / (g · FW)      (digital gain, seperti auto-exposure kamera ponsel)
    dark: y_lin = e / FW            (tanpa kompensasi)
    y     = round(255 · clip(y_lin, 0, 1)^(1/γ))
Penyederhanaan yang harus ditulis di Limitations: noise independen per kanal
(tanpa mosaik Bayer/demosaicing), tanpa pipeline ISP (denoise, tone mapping), gamma 2.2
sebagai pendekatan kurva sRGB. Rujukan pemodelan: Foi et al., IEEE TIP 2008;
Brooks et al., CVPR 2019 ("unprocessing").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import image_rng


@dataclass(frozen=True)
class Condition:
    kind: str            # clean | gauss | lowlight
    level: float = 0.0   # σ (gauss) atau gain g (lowlight)
    mode: str = "iso"    # lowlight: iso | dark

    @property
    def name(self) -> str:
        if self.kind == "clean":
            return "clean"
        if self.kind == "gauss":
            return f"gauss_s{self.level:g}"
        return f"ll_{self.mode}_g{self.level:g}"

    @staticmethod
    def parse(name: str) -> "Condition":
        if name == "clean":
            return Condition("clean")
        if name.startswith("gauss_s"):
            return Condition("gauss", float(name[7:]))
        if name.startswith("ll_"):
            _, mode, g = name.split("_")
            return Condition("lowlight", float(g[1:]), mode)
        raise ValueError(name)


def add_gaussian(rgb: np.ndarray, sigma: float, rng) -> np.ndarray:
    if sigma <= 0:
        return rgb
    y = rgb.astype(np.float64) + rng.normal(0.0, sigma, rgb.shape)
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def simulate_lowlight(rgb: np.ndarray, gain: float, rng, full_well=1000.0,
                      read_noise=3.0, gamma=2.2, mode="iso") -> np.ndarray:
    x_lin = (rgb.astype(np.float64) / 255.0) ** gamma
    lam = gain * full_well * x_lin
    e = rng.poisson(lam).astype(np.float64) + rng.normal(0.0, read_noise, rgb.shape)
    denom = gain * full_well if mode == "iso" else full_well
    y_lin = np.clip(e / denom, 0.0, 1.0)
    return np.clip(np.round(255.0 * y_lin ** (1.0 / gamma)), 0, 255).astype(np.uint8)


def apply(rgb: np.ndarray, cond: Condition, cfg, image_id: str) -> np.ndarray:
    if cond.kind == "clean":
        return rgb
    rng = image_rng(cfg["seed"], image_id, cond.name)
    if cond.kind == "gauss":
        return add_gaussian(rgb, cond.level, rng)
    ll = cfg["degradation"]["lowlight"]
    return simulate_lowlight(rgb, cond.level, rng, ll["full_well"], ll["read_noise"],
                             ll["gamma"], cond.mode)
