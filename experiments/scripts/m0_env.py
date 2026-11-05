"""M0a — bekukan environment dan cek versi (C7).

Output: results/env/requirements_frozen.txt, results/env/environment.json
Gagal (exit 1) kalau versi SciPy tidak kompatibel dengan NumPy yang terpasang.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys

from _cli import parser, setup
from kgexp.common import results_dir


def main():
    args = parser(__doc__).parse_args()
    cfg = setup(args)
    out = results_dir(cfg, "env")
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out / "requirements_frozen.txt").write_text(frozen)
    import cv2, numpy, scipy, skimage, pandas  # noqa: E401
    info = {
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit-image": skimage.__version__,
        "opencv": cv2.__version__, "pandas": pandas.__version__,
        "opencv_contrib_ximgproc": hasattr(cv2, "ximgproc"),
        "cv2_threads": cv2.getNumThreads(),
    }
    try:
        import phasepack  # noqa: F401
        info["phasepack"] = True
    except ImportError:
        info["phasepack"] = False
    (out / "environment.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    np_major = int(numpy.__version__.split(".")[0])
    sp_minor = tuple(int(x) for x in scipy.__version__.split(".")[:2])
    if np_major >= 2 and sp_minor < (1, 13):
        print(f"[FAIL] SciPy {scipy.__version__} tidak mendukung NumPy {numpy.__version__}.")
        sys.exit(1)
    print(f"[OK] Tulis versi ini di bagian 4.4 naskah (C7): SciPy {scipy.__version__}")


if __name__ == "__main__":
    main()
