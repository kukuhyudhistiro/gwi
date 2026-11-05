# weights/

`zerodce_epoch99.npz` adalah konversi format bobot resmi Zero-DCE
(`Zero-DCE_code/snapshots/Epoch99.pth`, repo https://github.com/Li-Chongyi/Zero-DCE) ke NumPy,
dibuat dengan `tools/convert_zerodce.py`. Nilai bobotnya tidak diubah.

- Sumber: C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, R. Cong, "Zero-Reference Deep Curve
  Estimation for Low-Light Image Enhancement," CVPR 2020.
- Lisensi asli: academic research only, Creative Commons Attribution-NonCommercial 4.0
  International (CC BY-NC 4.0), sesuai README repo resmi. Pertahankan atribusi ini kalau file
  ikut dipublikasikan di repo GitHub atau Zenodo.
- Verifikasi: `tools/verify_zerodce_torch.py`. Selisih output dengan model PyTorch resmi
  ≤ 1 level abu-abu (uji 2026-11-04).
