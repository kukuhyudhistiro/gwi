# GWi + ODPS

Imaginary-only Gabor wavelet (GWi) with orientation-aware double-peak suppression (ODPS) for
edge-based boundary detection. This repository holds the code, configuration, and experiment
runbook for the GWi + ODPS manuscript (in preparation).

## Place in the research roadmap

| Step | Work | Contribution | Code |
|---|---|---|---|
| 1 | GWi + ODPS (this repository) | ODPS on top of a fixed GWi stage | `kukuhyudhistiro/gwi` |
| 2 | A-GWi, JESA 59(9):2495–2508, 2026, [doi:10.18280/jesa.590906](https://doi.org/10.18280/jesa.590906) | Per-pixel centre frequency and scale of the GWi kernel | `kukuhyudhistiro/localgradientactivity-` |

The GWi stage here is the static GWi baseline of the A-GWi article (Table 3 there), unchanged.
ODPS is the only addition, so any difference between GWi and GWi + ODPS comes from ODPS alone. Centre
frequency selection is the subject of the A-GWi article and is not tuned here.

| Stage | Setting |
|---|---|
| Preprocessing | ITU-R BT.601 grayscale, `cv2.equalizeHist`, divide by 255 |
| Kernel | imaginary (sine) Gabor, 7 × 7, wavelength 4, σ = 0.56 × wavelength, aspect 0.5, phase 0, not normalised |
| Orientations | 8, spaced 22.5°, maximum absolute response across orientations |
| Border | reflection padding (`cv2.BORDER_REFLECT`), also for the complex Gabor baseline |
| Output | min-max rescaling per image to [0, 1] |
| ODPS | non-maximum test along the argmax orientation at distance d = floor(wavelength / 2 + 0.5) = 2 |

`experiments/scripts/m0_paper2_regression.py` checks that this pipeline reproduces the GWi and GWC
rows of the A-GWi article on BSDS500 test and UDED.

An earlier version of this work, which used a 5 × 5 kernel with zero padding, is archived at
`kukuhyudhistiro/gwi-odps` (tag `v2.0.0-paper1`). The evaluation gates of the present experiments
were frozen before the first run with tag `prereg-b0-2026-11-05` on branch `ijies` of that
repository. The alignment of the GWi stage with the A-GWi article is a recorded deviation from that
protocol; see `experiments/CHANGELOG.md`, version 0.3.0.

## Layout

```
experiments/          Experiment package: config, modules, scripts, tests, runbook, changelog
  config.yaml         Every parameter and every pre-registered decision rule
  RUNBOOK.md          Step-by-step instructions (Indonesian), with run times and decision points
  kgexp/              Methods, degradation models, fast Berkeley evaluator, statistics
  scripts/            One script per module (M0, M0c, B0–B8, M9, C6)
  results/            Summary tables committed per module; count caches are not tracked
tools/scripts/        Method modules (gabor_core, nms_odps, baselines) and the reference evaluator 04c
eval_results/k5/      ODS summary of the earlier version, used as the M0 regression reference
data/                 Datasets, not tracked; see data/README.md
```

## Setup

Python 3.13 on Windows was used. The environment is pinned to the versions that produced the
earlier tables:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r experiments\requirements-experiments.txt
.\.venv\Scripts\python.exe -m pip install numpy==2.2.6 pandas==2.3.3
.\.venv\Scripts\python.exe -m pytest experiments\tests -q
```

`experiments/RUNBOOK.md` gives the order of the modules, the command for each one, and how long
each takes. Every script resolves its paths from its own location, so it can be started from any
folder.

## Evaluation protocol

Berkeley protocol with 99 thresholds, a matching tolerance of 0.0075 times the image diagonal,
thinning of each thresholded map, greedy one-to-one matching per annotator, and dataset-level count
accumulation for ODS, OIS, and AP. The fast evaluator in `experiments/kgexp/fast_eval.py` gives
counts identical to `tools/scripts/04c_evaluate_bsds_v2.py`; `experiments/scripts/m0_verify_fast_eval.py`
checks this per image and per threshold.

## License

MIT, see `LICENSE`. The datasets keep their own licenses. `experiments/weights/zerodce_epoch99.npz`
is a format conversion of the official Zero-DCE weights (Guo et al., CVPR 2020), released for
academic research only under CC BY-NC 4.0; see `experiments/weights/README.md`.

## Citation

See `CITATION.cff`. If you use the GWi stage, please also cite the A-GWi article above.
