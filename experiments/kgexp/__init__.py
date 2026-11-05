"""kgexp — paket eksperimen revisi GWi+ODPS (M0, B0–B8, M9).

Dipasang di <repo>/experiments/. Memakai ulang modul inti di <repo>/src
(gabor_core, nms_odps, baselines) dan evaluator <repo>/scripts/04c_evaluate_bsds_v2.py
agar angka tetap identik dengan protokol naskah.
"""
import os

# Single-thread untuk BLAS/OpenCV di semua proses (timing dan paritas numerik).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

__version__ = "0.1.0"
