# data/

The datasets are not tracked. Obtain each one from its original source under its own license and
place it in this layout, which is what `experiments/config.yaml` reads:

```
data/BSDS500/images/{train,val,test}/*.jpg     data/BSDS500/groundTruth/{train,val,test}/*.mat   (200/100/200)
data/BIPED/edges/imgs/test/*.jpg               data/BIPED/edge_maps/test/*.png                   (50)
data/BIPED/edges/imgs/train/rgbr/real/*.jpg    data/BIPED/edges/edge_maps/train/rgbr/real/*.png  (optional, B1 stability check only)
data/UDED/imgs/*.jpg                           data/UDED/gt/*.png                                (30)
```

- BSDS500: Arbelaez, Maire, Fowlkes, Malik, *Contour detection and hierarchical image
  segmentation*, IEEE TPAMI 33(5), 2011. Train and val are used for tuning; test is used only for
  reporting.
- BIPED v2: Soria, Riba, Sappa, *Dense extreme inception network*, WACV 2020. The ground truth of
  the test split is read from `edge_maps/test`, not from `edges/edge_maps/test`.
- UDED: Soria, Pomboza-Junez, Sappa, *LDC: Lightweight dense CNN for edge detection*, IEEE Access
  10, 2022. All 30 images are used; UDED is never used for tuning.

Check the number of image and ground-truth pairs (PowerShell, from the repository root):

```powershell
.\.venv\Scripts\python.exe -W ignore -c "import sys; sys.path.insert(0, 'experiments'); from kgexp.common import load_config; from kgexp.data import list_samples; c=load_config(); [print(d, s, len(list_samples(c, d, s))) for d, s in [('BSDS500','train'),('BSDS500','val'),('BSDS500','test'),('BIPED','test'),('UDED','test')]]"
```
