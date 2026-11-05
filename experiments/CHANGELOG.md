# CHANGELOG — experiments/

Catat setiap perubahan pada `config.yaml` (terutama `gates`) dan pada evaluator, beserta tanggal
dan alasannya. Perubahan gate SETELAH modul terkait dijalankan pada data asli harus dilaporkan di naskah.

## 0.3.0 — 2026-11-05 (tahap GWi diselaraskan dengan Paper 2; DEVIASI dari pra-registrasi)

Wajib dilaporkan di naskah sebagai deviasi dari protokol `prereg-b0-2026-11-05`.

- **Alasan.** Paper 2 sudah terbit: Yudhistiro, Basuki, Rijati (2026), *Adaptive Imaginary Gabor
  Wavelet*, JESA 59(9):2495–2508, doi:10.18280/jesa.590906. Dalam roadmap penelitian, paper
  GWi+ODPS adalah tahap sebelum Paper 2. Karena itu tahap GWi di sini harus sama dengan GWi statis
  Paper 2, dan ODPS menjadi satu-satunya tambahan. Keputusan penulis, 2026-11-05.
- **GWi** (`methods_default.gabor`):
  - k = 5 → 7. λ = 4, σ = 0.56·λ, γ = 0.5, dan N = 8 tetap.
  - Border `constant` (zero padding) → `reflect` (`cv2.BORDER_REFLECT`), sesuai Paper 2
    Sec. 3.5 dan Tabel 2–3.
  - GWC dan Canny_matched mengikuti kernel dan border yang sama.
- **Preprocessing** (`preproc_default`): `he` (HE versi PIL) → `he_cv2` (`cv2.equalizeHist`),
  sesuai Paper 2 Eq. 1–2. `gates.b0_pilot.preproc` ikut berubah. Isi aturan gate tidak berubah.
- **B1 menjadi analisis sensitivitas.** `tuning.gabor_select: false` membuat GWi tetap di
  konfigurasi Paper 2. B1 tidak lagi menulis `selected_config.json` untuk GWi.
- **Hasil lama diarsipkan.** B0 dan B1 yang dijalankan pada 2026-11-05 dengan konfigurasi lama
  (k = 5, λ = 4, zero padding, HE PIL) dipindahkan ke
  `results/superseded_0.2.1_k5_zero-padding/`. Ringkasannya:
  - B0: NO-GO.
  - B1: terbaik k = 5, λ = 6. ODS test GWi 0.4725, GWi+ODPS 0.4917.
- **M0 tetap.** Regresi terhadap Tabel 5–7 naskah DSP tetap memakai `methods_legacy.gabor` dan
  HE PIL, sehingga kunci cache M0 tidak berubah.
- **Baru: `scripts/m0_paper2_regression.py` (M0c).** Skrip ini mereproduksi ODS GWi, GWC, Sobel,
  dan Canny Paper 2 (Tabel 5–6) di BSDS500 test dan UDED.
- `kgexp/methods.py`:
  - Preprocessing baru `he_cv2`.
  - Kunci spesifikasi `border` untuk GWi, GWi+ODPS, dan GWC. Tanpa kunci ini, perilakunya tetap
    zero padding seperti naskah DSP.
  - `gwc_magnitude`.
  - `gabor_from` mengikuti `tuning.gabor_select`.

## 0.2.1 — 2026-11-05 (penyesuaian path, sebelum modul apa pun dijalankan; gate tidak berubah)

- Paket dipasang di repo `gwi-odps` (branch `ijies`) sebagai `experiments/`. `kgexp/common.py` membaca
  modul legacy dari `src/` + `scripts/` (layout GitHub), atau dari `tools/scripts/` kalau `src/` tidak ada
  (folder kerja lokal). `src/{baselines,gabor_core,nms_odps}.py` di GitHub dan `tools/scripts/` lokal
  hanya berbeda di docstring; `02_run_unified.py`, `04c_evaluate_bsds_v2.py`, dan
  `eval_results/k5/ods_summary.csv` identik.
- `config.yaml`: GT BIPED test → `edge_maps/test`, sebelumnya `edges/edge_maps/test`. Ini mengikuti
  `src/dataset_io.py` (pipeline Tabel 5–7) dan data asli; `data/README.md` di repo yang keliru.
- Environment disamakan dengan saat Tabel 5–7 dibuat: NumPy 2.4.6 → 2.2.6, pandas 3.0.3 → 2.3.3.
- `RUNBOOK.md`: lokasi kerja tunggal `D:\Paper GWi ODPS\project_GWi_ODPS_v2`; semua perintah ditulis
  untuk PowerShell dengan `$PY`/`$EXP` dan path absolut; log run panjang ke `results\logs\`;
  commit/push ke GitHub lewat clone sementara di `%TEMP%` yang dihapus setelah push (bagian 2.4).
- M0 di mesin asal Tabel 5–7 (2026-11-05): `m0_verify_fast_eval --n 5` 135 pasangan, selisih 0,
  speedup median 7.2×; regresi 24/24 identik (|Δ| ODS/OIS/AP ≤ 6e-17). Bukti: `results/m0/`, `results/env/`.

## 0.2.0 — 2026-11-04 (sebelum ada modul yang dijalankan pada data asli, kecuali M0)

- **Evaluator cepat** (`kgexp/fast_eval.py`, `eval.engine: fast`). Count identik dengan
  `04c_evaluate_bsds_v2.py`, median 6.4× lebih cepat (2–20× per metode). Diuji pada data sintetis
  (`tests/test_fast_eval.py`) dan BSDS500 asli (`scripts/m0_verify_fast_eval.py`).
- **Gate B0 dan B7b direvisi sebelum dijalankan.** Pembanding pipeline sekarang dipisah per mode:
  - iso → `BM3D_Canny` (denoise lalu Canny)
  - dark → `ZeroDCE_Canny` (enhance lalu Canny)
  
  Alasan: pada mode iso, citra sudah dikompensasi gain sehingga terang. Zero-DCE yang mencerahkan
  citra bukan pembanding yang wajar di situ; uji pada BSDS500 val menaikkan rerata intensitas
  citra bersih dari 90 ke 145. Keputusan B0 diambil dari mode iso, dan mode dark menjadi sekunder.
- B0 memakai σ Canny terpilih di kondisi clean untuk pipeline BM3D/Zero-DCE. Sebelumnya dipakai
  σ default 1.4.
- B7b: BM3D_Canny menjadi bawaan (`--no-bm3d` untuk uji cepat), dan `b7b_framing` diturunkan
  otomatis dari keputusan iso dan dark.
- `tools/verify_zerodce_torch.py`: Zero-DCE NumPy vs model PyTorch resmi, selisih maksimum
  1 level abu-abu pada 6 uji.
- M0 di container uji (BSDS500 test, 8 metode): ODS sama dengan `ods_summary.csv` sampai
  3 desimal; desimal ke-4 berbeda (maks. 0.00015). Peta prediksi identik dengan
  `02_run_unified.py` di environment yang sama, jadi selisihnya berasal dari versi library.
  Regresi 4 desimal harus dijalankan di mesin asal Tabel 5–7.

## 0.1.0 — 2026-11-04

- Paket awal: M0, B0–B8, M9, C6, cache count, bootstrap, model low-light Poisson–Gaussian,
  Zero-DCE NumPy, ODPS tervektorisasi.
