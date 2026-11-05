# Runbook eksperimen Paper 1 (GWi+ODPS → IJIES)

Cakupan: BSDS500, BIPED v2, UDED, dan versi terdegradasi BSDS500 (noise Gaussian, low-light
sintetis). Data jagung RGB/UV **tidak** dipakai di Paper 1; data itu untuk Paper 3.

**Tahap GWi = GWi statis Paper 2** (sejak 0.3.0, lihat `CHANGELOG.md`). Dalam roadmap penelitian,
paper ini adalah tahap sebelum Paper 2 (Yudhistiro, Basuki, Rijati, *Adaptive Imaginary Gabor
Wavelet*, JESA 59(9):2495–2508, 2026, doi:10.18280/jesa.590906). Karena itu GWi di sini persis sama
dengan baseline GWi Paper 2 (Tabel 3), dan ODPS menjadi satu-satunya tambahan:

| Tahap | Nilai |
|---|---|
| Preprocessing | BT.601 → `cv2.equalizeHist` → /255 (`preproc_default: he_cv2`) |
| Kernel | imaginary-only, k = 7, λ = 4, σ = 0.56·λ, γ = 0.5, ψ = 0, tanpa normalisasi kernel |
| Orientasi | N = 8 (0°, 22.5°, …, 157.5°), max \|respons\| antar-orientasi |
| Border | reflection padding (`cv2.BORDER_REFLECT`), juga untuk GWC |
| Output | min-max per citra ke [0, 1] |
| ODPS | d = floor(λ/2 + 0.5) = 2, arah normal dari argmax orientasi |

Konfigurasi naskah DSP (k = 5, zero padding, HE PIL) hanya dipakai untuk regresi M0
(`methods_legacy` di `config.yaml`).

Semua parameter dan aturan keputusan ada di `config.yaml`. Setiap modul membaca file itu,
menyimpan count per citra per threshold di `results/counts/`, dan menulis tabel ke
`results/<modul>/`. Modul berikutnya membaca cache, jadi menjalankan ulang hanya menghitung
yang berubah.

## 0. Lokasi kerja dan cara menjalankan perintah

Semua pekerjaan dilakukan di **satu folder**:

| Isi | Path |
|---|---|
| Root project | `D:\Paper GWi ODPS\project_GWi_ODPS_v2` |
| Paket eksperimen (folder ini) | `D:\Paper GWi ODPS\project_GWi_ODPS_v2\experiments` |
| Python (venv) | `D:\Paper GWi ODPS\project_GWi_ODPS_v2\.venv\Scripts\python.exe` |
| Modul metode + evaluator 04c | `D:\Paper GWi ODPS\project_GWi_ODPS_v2\tools\scripts` |
| Data | `D:\Paper GWi ODPS\project_GWi_ODPS_v2\data` |
| Acuan regresi Tabel 5–7 | `D:\Paper GWi ODPS\project_GWi_ODPS_v2\eval_results\k5\ods_summary.csv` |
| Acuan regresi GWi Paper 2 | konstanta di `scripts\m0_paper2_regression.py` (dari repo `localgradientactivity-`) |
| Hasil | `D:\Paper GWi ODPS\project_GWi_ODPS_v2\experiments\results` |
| Log run panjang | `D:\Paper GWi ODPS\project_GWi_ODPS_v2\experiments\results\logs` |
| Repo GitHub paper ini | https://github.com/kukuhyudhistiro/gwi (branch `main`) |
| Versi naskah DSP (arsip) | https://github.com/kukuhyudhistiro/gwi-odps, tag `v2.0.0-paper1`; tag `prereg-b0-2026-11-05` di branch `ijies` |

Semua perintah di runbook ini ditulis untuk **PowerShell** (terminal default Windows/VS Code).
Buka PowerShell, lalu jalankan dua baris ini **sekali di awal setiap sesi terminal**:

```powershell
$PY  = "D:\Paper GWi ODPS\project_GWi_ODPS_v2\.venv\Scripts\python.exe"
$EXP = "D:\Paper GWi ODPS\project_GWi_ODPS_v2\experiments"
```

Setelah itu setiap perintah berbentuk `& $PY "$EXP\scripts\<script>.py" [argumen]`.

- `&` wajib di PowerShell untuk menjalankan program yang path-nya diberi tanda kutip.
  Tanpa `&` muncul error `Unexpected token`.
- Path script ditulis lengkap, jadi perintah bisa dijalankan dari folder mana pun. Setiap script
  menghitung lokasi `config.yaml`, `data\`, `tools\scripts\`, dan `results\` dari lokasi file
  script itu sendiri, bukan dari folder aktif terminal.
- `kgexp/common.py` membaca modul metode dari `tools\scripts\`. Repo `gwi` memakai layout yang
  sama. Di repo lama `gwi-odps`, modul dibaca otomatis dari `src\` + `scripts\`.
- Run panjang (lebih dari ±10 menit): simpan log ke file supaya progres bisa dicek dan hasilnya tidak
  hilang kalau jendela terminal tertutup:

  ```powershell
  & $PY "$EXP\scripts\m0_cache.py" *> "$EXP\results\logs\m0_cache.log"
  Get-Content "$EXP\results\logs\m0_cache.log" -Tail 20 -Wait   # di jendela lain, untuk memantau
  ```

  Buat folder log sekali: `New-Item -ItemType Directory -Force "$EXP\results\logs"`.
- Sinkronisasi Google Drive harus **dimatikan** selama eksperimen berjalan (cache berisi ribuan
  file kecil).

## 1. Alur dan titik keputusan

```
M0  env → verifikasi evaluator → cache + regresi Tabel 5–7
M0c reproduksi GWi/GWC Paper 2 (BSDS500 test, UDED) ──► harus LULUS sebelum B0
 │
 ├── pra-registrasi: tag prereg-b0 (gwi-odps); deviasi 0.3.0 dicatat di CHANGELOG
 │
B0  pilot low-light (BSDS500 val) ──► GO / NO-GO  (menentukan B7b dan arah penulisan)
 │
B1  sensitivitas k × λ (train+val)   ──► GWi tetap konfigurasi Paper 2, tidak dipilih ulang
B2  tuning baseline (train+val)      ──► GWC_tuned menutup > 50% gap? → lunakkan klaim
B3  CI bootstrap + Wilcoxon/Holm     ──► CI mencakup 0 → tidak boleh ditulis "meningkat"
B6  MAC per piksel (analitis)
B7  noise Gaussian (test)            ──► PASS → framing (b)
B7b low-light (test, hanya jika B0 = GO) ──► iso PASS → framing low-light
B8  ODPS tervektorisasi + runtime    ──► wording "faster than" / "comparable to" GWC
B4  DL ringan, protokol sama (peta dari repo resmi)
B5  kalibrasi evaluator              ──► offset > 0.05 → pertimbangkan ganti evaluator, rerun
 │
M9  rerun final dari satu perintah → C6 audit angka naskah
```

Matriks framing (diputuskan otomatis dari `results\b7\decision.json`):

| B7b iso | B7 (Gaussian) | Framing naskah |
|---|---|---|
| PASS | apa pun | Low-light. B7 sebagai pendukung. Kalau B7b dark juga PASS, klaim mencakup tanpa kompensasi gain. |
| FAIL / SKIPPED | PASS | (b) keunggulan di noise Gaussian σ ≥ 20 |
| FAIL / SKIPPED | FAIL | (a) kontribusi dalam keluarga Gabor; akui Canny lebih unggul |

Klaim "50% konvolusi" selalu berlaku terhadap GWC saja (B6), dan selalu disertai runtime terukur (B8).

## 2. Persiapan

### 2.1 Environment (sudah dilakukan 2026-11-05)

Environment harus sama dengan saat Tabel 5–7 dibuat (`..\env_info.txt`, `..\requirements.txt`):
Python 3.13.13, NumPy 2.2.6, SciPy 1.15.3, scikit-image 0.25.2, OpenCV(-contrib) 4.13.0.92,
pandas 2.3.3, Pillow 12.2.0.

```powershell
& $PY -m pip install -r "D:\Paper GWi ODPS\project_GWi_ODPS_v2\requirements.txt" -r "$EXP\requirements-experiments.txt"
& $PY -m pip install numpy==2.2.6 pandas==2.3.3      # pin ulang kalau ada paket yang menaikkan versi
& $PY -m pip check                                  # harus: No broken requirements found
& $PY -m pytest "$EXP\tests" -q -p no:cacheprovider # 11 uji harus lolos
```

- OpenCV terpasang sebagai `opencv-contrib-python` 4.13.0.92 (dibutuhkan ED, `cv2.ximgproc`),
  menggantikan `opencv-python` dengan versi yang sama.
- Peringatan `pyfftw could not be imported` boleh diabaikan: Phase Congruency memakai FFT bawaan
  (lebih lambat, hasil sama).
- Jangan menjalankan `pip install --upgrade` pada venv ini selama eksperimen berlangsung.

### 2.2 Bobot Zero-DCE (sudah ada: `$EXP\weights\zerodce_epoch99.npz`)

Hanya perlu diulang kalau file .npz hilang:

```powershell
git clone --depth 1 https://github.com/Li-Chongyi/Zero-DCE "D:\Paper GWi ODPS\project_GWi_ODPS_v2\third_party\Zero-DCE"
& $PY "$EXP\tools\convert_zerodce.py" --pth "D:\Paper GWi ODPS\project_GWi_ODPS_v2\third_party\Zero-DCE\Zero-DCE_code\snapshots\Epoch99.pth"
# opsional, butuh PyTorch: cek kesetaraan dengan model resmi (lolos jika ≤ 1 level abu-abu)
& $PY "$EXP\tools\verify_zerodce_torch.py" --repo "D:\Paper GWi ODPS\project_GWi_ODPS_v2\third_party\Zero-DCE"
```

### 2.3 Data

`paths.data_root` = `..\data` relatif terhadap `experiments\`, yaitu
`D:\Paper GWi ODPS\project_GWi_ODPS_v2\data`. Layout yang dibaca `config.yaml`:

```
data\BSDS500\images\{train,val,test}\*.jpg      data\BSDS500\groundTruth\{train,val,test}\*.mat   (200/100/200)
data\BIPED\edges\imgs\test\*.jpg                data\BIPED\edge_maps\test\*.png                   (50)
data\BIPED\edges\imgs\train\rgbr\real\*.jpg     data\BIPED\edges\edge_maps\train\rgbr\real\*.png  (BELUM ADA)
data\UDED\imgs\*.jpg                            data\UDED\gt\*.png                                (30)
```

- GT BIPED test ada di `data\BIPED\edge_maps\test` (sama dengan `tools\scripts\dataset_io.py`,
  pipeline Tabel 5–7), bukan di `edges\edge_maps\test`.
- BIPED train hanya dipakai B1 untuk cek stabilitas. Selama belum diunduh, B1 otomatis melewatinya
  dengan peringatan `[WARN] BIPED train dilewati`; jalankan B1 dengan `--skip-biped` supaya jelas.
- File `desktop.ini` dari Google Drive diabaikan otomatis oleh pembaca data.

Cek cepat jumlah pasangan citra–GT:

```powershell
& $PY -W ignore -c "import sys; sys.path.insert(0, r'$EXP'); from kgexp.common import load_config; from kgexp.data import list_samples; c=load_config(); [print(d, s, len(list_samples(c, d, s))) for d, s in [('BSDS500','train'),('BSDS500','val'),('BSDS500','test'),('BIPED','test'),('UDED','test')]]"
```

### 2.4 Pra-registrasi dan commit ke GitHub

**Pra-registrasi (sudah dilakukan).** `config.yaml` dibekukan sebelum B0 dengan tag
`prereg-b0-2026-11-05` di repo `gwi-odps` (branch `ijies`, commit `b0e3adb`). Penyelarasan GWi
dengan Paper 2 (0.3.0) adalah deviasi dari protokol itu dan sudah dicatat di `CHANGELOG.md`. Kalau
sebuah gate terpaksa diubah lagi, catat di `CHANGELOG.md` beserta alasannya, dan laporkan
perubahannya di naskah.

**Commit dan push.** Repo paper ini adalah `kukuhyudhistiro/gwi` (branch `main`). Isinya
mencerminkan folder ini, kecuali data, cache, dan folder kerja lokal (`.gitignore`):

```
README.md  LICENSE  CITATION.cff  .gitignore  requirements.txt
data\README.md
eval_results\k5\ods_summary.csv                       acuan regresi M0
tools\scripts\{gabor_core,nms_odps,baselines,04c_evaluate_bsds_v2}.py
experiments\                                          tanpa results\, kecuali ringkasan yang dipilih
```

Folder `.git` di `project_GWi_ODPS_v2` rusak oleh Google Drive, jadi **jangan commit atau push
dari folder ini**. Untuk setiap commit, buat clone **sementara** di folder temp Windows, salin
dari folder ini, commit dan push, lalu hapus clone-nya:

```powershell
$ROOT = "D:\Paper GWi ODPS\project_GWi_ODPS_v2"
$TMP  = Join-Path $env:TEMP "gwi-sync"
git clone https://github.com/kukuhyudhistiro/gwi.git $TMP

# file tingkat repo
foreach ($f in "README.md","LICENSE","CITATION.cff",".gitignore","requirements.txt") { Copy-Item "$ROOT\$f" "$TMP\$f" -Force }
robocopy "$ROOT\data" "$TMP\data" README.md
robocopy "$ROOT\eval_results\k5" "$TMP\eval_results\k5" ods_summary.csv
robocopy "$ROOT\tools\scripts" "$TMP\tools\scripts" gabor_core.py nms_odps.py baselines.py 04c_evaluate_bsds_v2.py

# paket eksperimen tanpa cache, lalu ringkasan hasil yang ikut di-commit
robocopy "$EXP" "$TMP\experiments" /MIR /XD results __pycache__ .pytest_cache /XF desktop.ini
foreach ($m in "env","m0") { robocopy "$EXP\results\$m" "$TMP\experiments\results\$m" /MIR /XF desktop.ini }

git -C $TMP status --short                     # periksa daftar file sebelum commit
git -C $TMP add -A
git -C $TMP commit -m "<pesan>"
git -C $TMP push origin main

Set-Location $env:USERPROFILE; Remove-Item -Recurse -Force $TMP   # hapus clone sementara
```

- `robocopy` dengan exit code 0–7 berarti berhasil (1 = ada file yang disalin).
- `/MIR` membuat isi tujuan sama persis dengan sumber, jadi sumbernya selalu folder ini.
- Untuk commit hasil modul berikutnya, tambahkan nama foldernya ke daftar `foreach ($m in ...)`,
  mis. `"env","m0","b0","b1"`. `results\counts\`, `results\maps\`, dan `results\superseded_*`
  tidak di-commit.

## 3. Modul

Perkiraan waktu komputasi di bawah berasal dari M0 di container uji: 1600 pasangan
citra × metode BSDS500 dalam 51 menit dengan 2 proses, atau sekitar 3.8 CPU-detik per pasangan
(evaluasi 99 threshold + metode). Angkanya lalu dibagi 8 untuk `n_jobs: 8`, dengan asumsi kecepatan
per core sama. Citra BIPED (1280×720) dihitung 6× biaya BSDS. BM3D menambah ±7 detik dan
Zero-DCE NumPy ±5 detik per citra. Ukur ulang di mesin ini (i5-14400, 10 core / 16 thread):
M0 mencetak progres, dan `m0_verify_fast_eval.py` mencetak waktu per citra.

| Modul | Pasangan citra × metode (setara BSDS) | CPU-jam | Wall-clock, `n_jobs: 8` |
|---|---|---|---|
| M0 | ±4 200 | ±4.5 | ±35 menit |
| B0 (pre = he_cv2 dan none) | 12 600 + 1 600 enhancement | ±18 | ±2.2 jam (±1.1 jam dengan `--preproc he_cv2`) |
| B1 (+ BIPED train) | 9 600 + ±38 000 | ±50 | ±6 jam (±1.3 jam dengan `--skip-biped`) |
| B2 | ±12 900 | ±14 | ±2 jam |
| B3 | ±1 500 (sebagian dari cache) | ±1.5 | ±15 menit |
| B7 | ±7 200 | ±8 | ±1 jam |
| B7b | ±20 000 + 2 400 enhancement | ±25 | ±3 jam |
| **Total** | | **±120** | **±15 jam** (semalam ×2) |

B4 (inferensi DL), B5, B6, dan B8 tidak masuk tabel: B4 bergantung pada model, B6 analitis, dan B5/B8 hitungan kecil.

Argumen yang berlaku untuk semua script modul (`scripts\_cli.py`):

| Argumen | Fungsi |
|---|---|
| `--config <path>` | pakai config lain (default `$EXP\config.yaml`) |
| `--n-jobs <n>` | menimpa `eval.n_jobs` (default 8) |
| `--limit <n>` | batasi jumlah citra per split. **Hanya uji cepat, tidak boleh masuk naskah.** |
| `--force` | abaikan cache dan hitung ulang |

### M0 — environment, evaluator, cache (HK1)

```powershell
& $PY "$EXP\scripts\m0_env.py"                                                     # C7: bekukan versi
& $PY "$EXP\scripts\m0_verify_fast_eval.py" --n 5 *> "$EXP\results\logs\m0_verify.log"  # evaluator cepat vs 04c: selisih harus 0
& $PY "$EXP\scripts\m0_cache.py" *> "$EXP\results\logs\m0_cache.log"                   # cache 8 metode × 3 dataset + regresi Tabel 5–7
```

- Output: `results\env\`, `results\m0\`, `results\counts\`.
- `m0_verify_fast_eval.py` berjalan **satu proses** dan menghitung setiap pasangan dua kali
  (fast + 04c). Di mesin ini: ±40 detik per pasangan BSDS500, BIPED ±6× lebih lama; dengan `--n 5`
  × 3 dataset × 9 metode (135 pasangan) totalnya ±3–4 jam. `m0_cache.py` boleh dijalankan
  **paralel** di jendela PowerShell lain (verify hanya memakai 1 dari 16 thread). Kalau verify
  akhirnya gagal, set `engine: legacy` lalu jalankan ulang `m0_cache.py --force`.
- Log `m0_cache.log` baru ada setelah `m0_cache.py` dimulai.
- `engine: fast` di config memberi count yang identik dengan `04c_evaluate_bsds_v2.py`. Diuji per
  citra per threshold pada 27 pasangan citra × metode BSDS500 test (8 metode naskah + Canny
  hysteresis): selisih 0, median 6.4× lebih cepat (2× untuk GWC, 20× untuk ED).
  Kalau verifikasi gagal, set `engine: legacy` di `config.yaml`.
- Regresi lolos kalau ODS/OIS/AP sama dengan `..\eval_results\k5\ods_summary.csv` sampai 4 desimal
  (`--tol`, default 5e-5). Mesin ini adalah mesin yang menghasilkan Tabel 5–7 (`..\env_info.txt`,
  i5-14400, Windows 11), dan versi library sudah disamakan (bagian 2.1).
  Hasil di container uji (Linux, NumPy 2.5, SciPy 1.18) sebagai pembanding:
  - ODS sama sampai 3 desimal untuk kedelapan metode.
  - Desimal ke-4 berbeda untuk tujuh metode; |ΔODS| maksimum 0.00015 (LoG), Canny 0.00012.
  - Peta in-memory paket ini identik piksel-per-piksel dengan `02_run_unified.py` di container yang sama.

  Kalau regresi gagal di mesin ini, bandingkan `results\env\requirements_frozen.txt` dengan
  `..\requirements.txt`. Setelah itu ada dua pilihan: kembali ke versi itu, atau jalankan ulang
  semua tabel naskah di M9 dengan environment yang sudah dibekukan, lalu laporkan versinya.

### M0c — reproduksi GWi Paper 2 (wajib sebelum B0)

```powershell
& $PY "$EXP\scripts\m0_paper2_regression.py" *> "$EXP\results\logs\m0_paper2.log"   # ±10 menit
```

- Menghitung GWi, GWC, Sobel, dan Canny (magnitudo gradien, σ = 1.4) dengan konfigurasi
  `methods_default` di BSDS500 test (200) dan UDED (30). Hasilnya dibandingkan dengan ODS Tabel 5–6
  Paper 2 (`main_results_ods_summary.csv` di repo `localgradientactivity-`).
- Lulus kalau |ΔODS| ≤ 0.002 untuk GWi dan GWC di kedua dataset (`--tol`). Sobel dan Canny hanya
  dilaporkan sebagai kontrol preprocessing.
- Output: `results\m0\paper2_regression.{csv,md,json}`. Exit code 1 kalau gagal.

### B0 — pilot low-light (HK2–HK3)

Syarat: tag `prereg-b0` (bagian 2.4) dan M0c LULUS.

```powershell
& $PY "$EXP\scripts\b0_pilot.py" *> "$EXP\results\logs\b0.log"                # BSDS500 val, pre = he_cv2 dan none, mode iso dan dark
& $PY "$EXP\scripts\b0_pilot.py" --preproc he_cv2 *> "$EXP\results\logs\b0.log" # versi cepat, hanya protokol utama
```

- Data: BSDS500 **val** (100 citra). Test set tidak disentuh.
- Kondisi: clean, lalu low-light g ∈ {1, 0.1, 0.05} di dua mode:
  - `iso`: gain digital, seperti auto-exposure kamera ponsel.
  - `dark`: tanpa kompensasi gain.
- Model degradasi (`kgexp\degrade.py`), dihitung di domain linear:

  ```latex
  x_{lin} = (x/255)^{\gamma},\quad e = \mathrm{Poisson}(g\,F\,x_{lin}) + \mathcal{N}(0,\sigma_r^2),\quad
  y_{lin} = \begin{cases} e/(gF) & \text{iso} \\ e/F & \text{dark} \end{cases},\quad
  y = 255\,\mathrm{clip}(y_{lin})^{1/\gamma}
  ```

  dengan F = 1000 e⁻ (full well), σ_r = 3 e⁻, γ = 2.2.
- Pembanding:
  - Semua mode: `Canny_clean_selected` (σ dipilih di clean, bukan per kondisi) dan `Canny_matched` (σ disamakan dengan support kernel GWi; kontrol confounder skala).
  - Mode iso: `BM3D_Canny` (denoise lalu Canny).
  - Mode dark: `ZeroDCE_Canny` (enhance lalu Canny).
  - `Canny_oracle` (σ terbaik per kondisi) dilaporkan sebagai batas atas saja, tidak masuk gate.
- Metrik tambahan: `f_transfer`, yaitu F pada threshold ODS dari kondisi clean, dipakai tanpa
  diubah. Ini skenario perangkat yang tidak bisa menala ulang threshold per adegan.
- **Gate (pra-registrasi):** GO kalau ΔODS(GWi+ODPS − c) > 0 untuk semua pembanding mode iso pada
  g = 0.1 dan 0.05 (pre = he_cv2). Mode dark dan pre = none dilaporkan sebagai sensitivitas.
- Output: `results\b0\gate.json`, `scores|deltas|transfer.csv`, `plot_ods_vs_gain_{iso,dark}.png`.
- n = 100 hanya untuk menentukan arah. Jangan kutip angka B0 sebagai klaim di naskah.

### B1 — sensitivitas k × λ untuk GWi dan GWi+ODPS (HK3–HK4)

```powershell
& $PY "$EXP\scripts\b1_tune_gwi.py" --skip-biped *> "$EXP\results\logs\b1.log"   # selama BIPED train belum ada
& $PY "$EXP\scripts\b1_tune_gwi.py" *> "$EXP\results\logs\b1.log"                # setelah BIPED train diunduh
```

- Grid k ∈ {3, 5, 7, 9} × λ ∈ {3, 4, 5, 6} di BSDS500 train+val, dengan proses GWi Paper 2
  (reflect, HE OpenCV). BIPED train hanya untuk cek stabilitas.
- Dengan `tuning.gabor_select: false` (bawaan sejak 0.3.0), GWi **tidak** dipilih ulang. Konfigurasi
  utama tetap k = 7, λ = 4 (Paper 2), dan B1 hanya melaporkan posisinya di grid. Pemilihan
  frekuensi adalah topik Paper 2, jadi di naskah cukup dirujuk ke sana.
- Output: `results\b1\grid_*.csv`, `results\b1\test_row.csv` (konfigurasi terbaik di grid, untuk
  pelaporan), dan `results\b1\decision.json` (`mode: sensitivity_only`). `selected_config.json`
  tidak disentuh.

### B2 — tuning setara baseline (HK4–HK5)

```powershell
& $PY "$EXP\scripts\b2_tune_baselines.py" *> "$EXP\results\logs\b2.log"
& $PY "$EXP\scripts\b2_tune_baselines.py" --only GWC Canny                   # subset keluarga
```

- Metode: GWC (grid sama dengan B1), Canny penuh (NMS + hysteresis, low = 0.4·high, high di-sweep
  sebagai threshold ODS), Sobel, LoG, PC, ED.
- Gate: kalau GWC_tuned menutup lebih dari 50% selisihnya dengan GWi, klaim "GWi > GWC" dilunakkan
  (`results\b2\decision.json`).

### B3 — CI dan uji berpasangan (HK6)

```powershell
& $PY "$EXP\scripts\b3_stats.py"
& $PY "$EXP\scripts\b3_stats.py" --datasets BSDS500        # satu dataset
```

- Bootstrap berpasangan 1000× untuk ΔODS. Wilcoxon signed-rank pada F per citra dengan
  rank-biserial r, dan koreksi Holm untuk 9 uji (3 pasang × 3 dataset).
- `results\b3\claims_guard.json` memberi kata yang boleh dipakai untuk tiap perbandingan:
  "meningkat", "menurun", atau "tidak berbeda (CI mencakup 0)".

### B4 — DL ringan dengan protokol yang sama (HK7–HK10)

Inferensi dijalankan dari repo resmi tiap metode, di luar paket ini. Simpan peta probabilitas 8-bit
ke `$EXP\results\maps\<metode>\<dataset>\<image_id>.png` dengan resolusi asli dan tanpa NMS khusus.
Isi `b4.methods` di `config.yaml` beserta dataset training tiap bobot.

```powershell
& $PY "$EXP\scripts\b4_dl.py" --eval --methods TEED PiDiNet_tiny LDC
& $PY "$EXP\scripts\b4_dl.py" --profile --name TEED --model <adapter>:build --weights "<path\bobot.pth>"
& $PY "$EXP\scripts\b4_dl.py" --pareto
```

Pilih minimal satu metode terbitan 2024 atau sesudahnya yang kode dan bobot resminya tersedia,
dan cek ketersediaannya di HK7. Hasil BIPED dari bobot yang dilatih di BIPED train harus diberi
tanda. UDED menjadi uji cross-domain. `--profile` butuh PyTorch (belum terpasang di venv ini).

### B5 — kalibrasi evaluator (HK11)

```powershell
& $PY "$EXP\scripts\b5_calibration.py" --classical
```

- `greedy` vs `maxcard` (maximum-cardinality bipartite) untuk semua metode klasik.
- Offset terhadap angka resmi: isi `b5.references` di `config.yaml` dengan ODS yang dilaporkan paper
  asli. Ambil angkanya langsung dari paper atau situs resmi, jangan dari ingatan.
- Gate: offset > 0.05 → pertimbangkan mengganti evaluator utama dan rerun M9.

### B6 — kompleksitas analitis (HK12, ½ hari)

```powershell
& $PY "$EXP\scripts\b6_complexity.py"
```

MAC per piksel (`results\b6\ops_per_pixel.csv`). GWi = 0.5 × GWC untuk k dan N yang sama.
Rasio Canny terhadap GWC (k = 7) dibaca dari tabel B6. Kedua angka ini harus ditulis berdampingan.

### B7 / B7b — noise dan low-light pada test set (HK12–HK14)

```powershell
& $PY "$EXP\scripts\b7_noise.py" *> "$EXP\results\logs\b7.log"   # B7 Gaussian σ ∈ {0, 5, 10, 20, 30}; B7b otomatis jika B0 = GO
& $PY "$EXP\scripts\b7_noise.py" --skip-b7                        # hanya B7b
& $PY "$EXP\scripts\b7_noise.py" --skip-b7b                       # hanya B7
```

- Parameter final dari B1/B2, tanpa tuning ulang per level.
- B7b: g ∈ {1, 0.3, 0.1, 0.05, 0.02}, mode iso dan dark. BM3D_Canny dan ZeroDCE_Canny memakai
  Canny yang sama dengan Canny_tuned. `--no-bm3d` hanya untuk uji cepat (gate iso jadi INCOMPLETE).
  `--force-b7b` menjalankan B7b walaupun B0 = NO-GO (hanya untuk dilaporkan sebagai sensitivitas).
- Gate B7: ΔODS(GWi+ODPS − Canny_tuned) > 0 **dan** CI_lo > 0 pada σ = 20 dan 30.
- Gate B7b per mode: ΔODS > 0 **dan** CI_lo > 0 terhadap semua pembanding mode itu, pada
  g = 0.1, 0.05, dan 0.02.
- Output: `results\b7\decision.json`, berisi keputusan B7, B7b per mode, dan `b7b_framing`.

### B8 — ODPS tervektorisasi dan runtime (HK15)

```powershell
& $PY "$EXP\scripts\b8_odps.py"
```

- Kesetaraan bit-per-bit ODPS lama vs cepat pada 10 citra; kalau gagal, exit 1.
- Runtime dengan 1 thread, 3 warm-up, median dari 3 putaran. Tutup aplikasi berat lain selama B8.
- Hasilnya menentukan wording abstrak: "faster than complex Gabor" atau "comparable".

### M9 — rerun final dan audit (HK16–HK17)

```powershell
& $PY "$EXP\scripts\m9_run_all.py" *> "$EXP\results\logs\m9.log"   # semua modul berurutan, berhenti kalau ada yang gagal
& $PY "$EXP\scripts\m9_run_all.py" --from b3                        # lanjut dari modul tertentu
& $PY "$EXP\scripts\m9_run_all.py" --only b0 b1                     # modul tertentu saja
& $PY "$EXP\scripts\m9_run_all.py" --extra --n-jobs 8               # argumen bersama untuk semua modul
& $PY "$EXP\scripts\c6_audit.py" --manuscript "D:\Paper GWi ODPS\project_GWi_ODPS_v2\manuscript\<draf_ijies>.docx"
```

- M9 memanggil setiap script dengan Python yang sama (`$PY`) dan path absolut, jadi tidak
  bergantung pada folder aktif terminal.
- `--extra` diteruskan ke **semua** modul; gunakan hanya argumen bersama (`--limit`, `--n-jobs`,
  `--force`). Argumen khusus modul (mis. `--skip-biped`) dijalankan per modul dengan `--only`.
- Daftarkan setiap angka di abstrak, teks, dan kesimpulan ke `$EXP\claims.yaml`. C6 menghitung
  ulang setiap angka dari CSV dan memeriksa bahwa string-nya ada di naskah.

## 4. Jadwal hari kerja (mulai Kamis 5 Nov 2026)

| HK | Tanggal | Modul | Bergantung pada |
|---|---|---|---|
| 1 | Kam 5 Nov | M0, pra-registrasi (2.4) | — |
| 2 | Jum 6 Nov | B0 (komputasi pilot) | M0, tag prereg-b0 |
| 3 | Sen 9 Nov | B0 (keputusan GO/NO-GO), mulai B1 | B0 |
| 4 | Sel 10 Nov | B1 selesai, mulai B2 | M0 |
| 5 | Rab 11 Nov | B2 | B1 |
| 6 | Kam 12 Nov | B3 | B1, B2 |
| 7 | Jum 13 Nov | B4: pilih metode, cek kode dan bobot | M0 |
| 8 | Sen 16 Nov | B4: inferensi BSDS500, BIPED | HK7 |
| 9 | Sel 17 Nov | B4: inferensi UDED, runtime | HK8 |
| 10 | Rab 18 Nov | B4: evaluasi, FLOPs, Pareto | HK9 |
| 11 | Kam 19 Nov | B5 | B4 |
| 12 | Jum 20 Nov | B6, B7 | B1, B2, B3 |
| 13 | Sen 23 Nov | B7b (iso, dark) | B0 = GO, B1, B2 |
| 14 | Sel 24 Nov | B7b keputusan framing | HK13 |
| 15 | Rab 25 Nov | B8 | B1 |
| 16 | Kam 26 Nov | M9 rerun semua tabel dan gambar | B1–B8 |
| 17 | Jum 27 Nov | C6 audit, update repo (2.4), tag rilis | HK16 |
| 18 | Sen 30 Nov | Buffer | — |

Jalur kritis: M0 → B0 → B1 → B2 → B3 → B7 → B7b → M9. Komputasi B1, B2, dan B7b bisa dijalankan
semalam. Kalau B0 = NO-GO, HK13–14 kosong, jadwal maju 2 hari, dan B7 kembali menentukan framing.

## 5. Keterbatasan simulasi yang wajib ditulis di naskah

- Noise independen per kanal: tanpa mosaik Bayer, demosaicing, atau pipeline ISP (denoise
  internal, tone mapping). Gamma 2.2 hanya pendekatan kurva sRGB [Foi et al., IEEE TIP 2008;
  Brooks et al., CVPR 2019].
- Tidak ada citra low-light nyata yang punya GT edge dari anotasi manusia. Klaim harus dibatasi ke
  "simulated low-light capture on BSDS500".
- BM3D memakai σ tunggal hasil estimasi. Varian dengan variance-stabilizing transform untuk noise
  Poisson–Gaussian belum diuji.
- HE (protokol naskah) sudah menormalkan kontras global. Analisis `pre = none` di B0 menunjukkan
  seberapa besar hasil bergantung pada HE.
- Zero-DCE versi NumPy hanya dipakai untuk akurasi. Untuk latency, ukur versi PyTorch resmi.

## 6. Aturan kerja

- Kerja dan eksekusi hanya di `D:\Paper GWi ODPS\project_GWi_ODPS_v2`. Commit/push ke GitHub
  (`kukuhyudhistiro/gwi`) lewat clone sementara (bagian 2.4) yang langsung dihapus setelah push;
  jangan menjalankan eksperimen di sana. Repo `gwi-odps` adalah arsip versi DSP dan tidak diubah lagi.
- Hasil dengan `--limit` hanya untuk uji cepat dan tidak boleh masuk naskah.
- Jangan ubah `gates` setelah modul yang memakainya dijalankan pada data asli.
- `tests\config_synth.yaml` dan `tools\make_synthetic_data.py` hanya untuk smoke test pipeline.
