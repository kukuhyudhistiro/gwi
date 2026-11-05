"""M9 — jalankan seluruh modul berurutan dari satu perintah, berhenti di setiap gate.

Urutan (jalur kritis): M0 (env, verifikasi evaluator, cache+regresi) → B0 → B1 → B2 → B3 → B6
                       → B7(+B7b) → B8 → B4(eval+pareto) → B5
    python scripts/m9_run_all.py                  # semua
    python scripts/m9_run_all.py --from b3        # lanjut dari B3
    python scripts/m9_run_all.py --only b0        # satu modul
Setiap modul memakai cache, jadi menjalankan ulang hanya menghitung yang berubah.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("m0env", ["m0_env.py"]),
    ("m0verify", ["m0_verify_fast_eval.py", "--n", "3"]),
    ("m0", ["m0_cache.py"]),
    ("b0", ["b0_pilot.py"]),
    ("b1", ["b1_tune_gwi.py"]),
    ("b2", ["b2_tune_baselines.py"]),
    ("b3", ["b3_stats.py"]),
    ("b6", ["b6_complexity.py"]),
    ("b7", ["b7_noise.py"]),
    ("b8", ["b8_odps.py"]),
    ("b4", ["b4_dl.py", "--eval", "--pareto"]),
    ("b5", ["b5_calibration.py", "--classical"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="argumen tambahan untuk semua modul, mis. --extra --limit 5")
    a = ap.parse_args()
    names = [n for n, _ in STAGES]
    sel = names[names.index(a.start):] if a.start else names
    if a.only:
        sel = [n for n in names if n in a.only]
    for n, cmd in STAGES:
        if n not in sel:
            continue
        print(f"\n{'=' * 70}\n  {n.upper()}: {' '.join(cmd)}\n{'=' * 70}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:], *a.extra])
        if r.returncode != 0:
            print(f"[STOP] {n} gagal (exit {r.returncode}). Perbaiki dulu sebelum lanjut.")
            sys.exit(r.returncode)
    print("\n[OK] Semua modul selesai. Jalankan c6_audit.py terhadap draf naskah.")


if __name__ == "__main__":
    main()
