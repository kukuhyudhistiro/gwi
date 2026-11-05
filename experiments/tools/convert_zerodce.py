"""Konversi bobot resmi Zero-DCE (Epoch99.pth, format serialisasi lama PyTorch) ke .npz
TANPA PyTorch. Unpickler dibatasi: hanya OrderedDict, FloatStorage, dan
torch._utils._rebuild_tensor_v2 yang diizinkan (kelas lain ditolak).

    git clone --depth 1 https://github.com/Li-Chongyi/Zero-DCE
    python tools/convert_zerodce.py --pth Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth
"""
from __future__ import annotations

import argparse
import collections
import pickle
import struct
from pathlib import Path

import numpy as np

DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
      "HalfStorage": np.float16}


def load_legacy_pth(path: Path) -> dict:
    f = open(path, "rb")
    storages = {}

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return lambda st, off, size, stride, *a: ("T", st, off, size, stride)
            if mod == "torch" and name in DT:
                return name
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"kelas tidak diizinkan: {mod}.{name}")

        def persistent_load(self, pid):
            _, typ, key, _loc, size = pid[:5]
            storages.setdefault(key, [typ, size, None])
            return key

    for _ in range(3):           # magic number, protocol version, sys info
        pickle.load(f)
    obj = U(f).load()
    keys = pickle.load(f)
    for k in keys:
        n = struct.unpack("<q", f.read(8))[0]
        dt = np.dtype(DT[storages[k][0]])
        storages[k][2] = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
    out = {}
    for name, (_, st, off, size, stride) in obj.items():
        a = storages[st][2]
        out[name] = np.lib.stride_tricks.as_strided(
            a[off:], shape=size, strides=[s * a.itemsize for s in stride]).copy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pth", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "weights" / "zerodce_epoch99.npz")
    a = ap.parse_args()
    w = load_legacy_pth(a.pth)
    expected = {f"e_conv{i}.{t}" for i in range(1, 8) for t in ("weight", "bias")}
    if set(w) != expected:
        raise SystemExit(f"kunci tidak sesuai arsitektur Zero-DCE: {sorted(set(w) ^ expected)}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **w)
    print(f"[OK] {a.out}  ({sum(v.size for v in w.values())} parameter)")


if __name__ == "__main__":
    main()
