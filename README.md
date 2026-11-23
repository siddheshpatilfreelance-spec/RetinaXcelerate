# RetinaXcelerate — A Scalable Deep Learning Pipeline for Diabetic Retinopathy Detection

Code for the paper *"RetinaXcelerate: A Scalable Deep Learning Pipeline for Diabetic
Retinopathy Detection"* (Patil, Kachare, Patel — SAKEC).

## Quick start (web app)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py          # then visit http://localhost:8000
```

## Paper pipeline (default) — paper → code

| Paper | Code |
|---|---|
| §III-A Checkpoint-agnostic abstraction, label remap (Eq. 1) | `src/hf_model.py` — `Member`, `EnsembleDR` |
| §III-B CCER (Eq. 2), dual-path Φ with α = 0.65 (Eq. 3), Algorithm 1 | `EnsembleDR.predict`, `_member_probs` |
| §III-C Circular crop (τ = 15), Ben Graham (σ = 10, Eq. 4) | `circle_crop`, `ben_graham` |
| §III-D Attention Rollout (Eq. 5–6) | `EnsembleDR.attention` |
| §IV-A rafalosa (primary, w = 1.0) + Kontawat (w = 1.0), ~172 M params | `MEMBERS` in `src/hf_model.py` |
| §IV-B `GET /api/info`, `POST /api/predict`, `GET /api/sample/{id}` | `server.py` |
| §IV-C 5-tab SPA, 7-image Wikimedia gallery (Table III) | `static/` |
| §V Tables IV, V, VI | `python -m src.paper_benchmark` → `results/paper_tables.md` |

Reproduction status:
- **Table IV** — every reported value reproduces exactly.
- **Table V** — rows 1, 2, 5 and 6 reproduce (Ben Graham flips the prediction to Moderate);
  "no TTA, no BG" gives No DR 0.35 and "+ TTA" gives Moderate 0.39 on this software stack
  (paper: No DR 0.41 / No DR 0.39).
- **Table VI** — median 0.28 s per inference here (MPS,
  PyTorch 2.15.0); the paper reports 2.11 s on its test machine.

## Enhanced mode (optional) — paper Future Work §VII-1

Tick **Enhanced mode** in the Predict tab, or pass `?mode=enhanced` to the API. Same CCER
aggregation (Eq. 2) with two extra members fine-tuned on APTOS 2019 + IDRiD; weights chosen
on a validation split. Paper mode stays the default.

```bash
python -m src.prepare_dr_data --variant hires --benchmark
python -m src.train_dr --arch vit    --epochs 10 --splits data/dr/splits_hires.csv --img-dir data/dr/images_hires
python -m src.train_dr --arch effnet --epochs 12 --splits data/dr/splits_hires.csv --img-dir data/dr/images_hires
python -m src.evaluate_dr
python -m src.evaluate_ccer      # -> checkpoints/ccer_weights.json, results/ccer_eval.json
```

Held-out results (never used for training or weight selection):

**IDRiD test (103 ophthalmologist-graded images)**

| Configuration | Accuracy | QWK | Referable sens. | Referable spec. |
|---|---|---|---|---|
| Paper CCER (rafalosa + Kontawat, equal weights) | 56.3% | 0.634 | 76.6% | 100.0% |
| Fine-tuned only (ViT + EfficientNet, val-tuned) | 62.1% | 0.719 | 92.2% | 74.4% |
| Enhanced CCER (4 members, val-tuned) | 62.1% | 0.716 | 90.6% | 74.4% |

**Messidor-2 (1,744 images, external)**

| Configuration | Accuracy | QWK | Referable sens. | Referable spec. |
|---|---|---|---|---|
| Paper CCER (rafalosa + Kontawat, equal weights) | 59.7% | 0.345 | 30.2% | 96.3% |
| Fine-tuned only (ViT + EfficientNet, val-tuned) | 65.8% | 0.573 | 46.8% | 99.1% |
| Enhanced CCER (4 members, val-tuned) | 64.6% | 0.535 | 43.1% | 99.2% |

CCER weights (validation): `{'rafalosa': 0.5, 'Kontawat': 0.0, 'vit': 1.5, 'effnet': 2.0}`. Limitations: Mild DR is rarely detected;
Enhanced mode grades the CMV out-of-distribution sample as DR; IDRiD test is small (±9 pts).

---

# RetinaXcelerate — M.Tech Project

End-to-end deep-learning pipeline for multi-label diagnosis of **46 retinal conditions**
(1 overall `Disease_Risk` + 45 fine-grained diseases) on the
[RFMiD dataset](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification/data),
with a fine-tuned ResNet50, full evaluation suite, Grad-CAM explainability, and a
Streamlit demo UI.

## Highlights

- **Two-stage transfer learning** — warm-up head, then discriminative fine-tune.
- **Class imbalance handled** — per-class `pos_weight` in `BCEWithLogitsLoss`.
- **Rich evaluation** — macro AUC, mAP, per-class precision/recall/F1, ROC & PR curves.
- **Grad-CAM** explainability overlay on the top predicted disease.
- **Multi-tab Streamlit app**: Predict · Model · Dataset · Results · About.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset

Requires a Kaggle API token at `~/.kaggle/kaggle.json`.

```bash
pip install kaggle
kaggle datasets download -d andrewmvd/retinal-disease-classification -p data --unzip
```

Expected layout after unzip:

```
data/
  Training_Set/Training_Set/Training/*.png
  Training_Set/Training_Set/RFMiD_Training_Labels.csv
  Evaluation_Set/Evaluation_Set/Validation/*.png
  Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv
  Test_Set/Test_Set/Test/*.png
  Test_Set/Test_Set/RFMiD_Testing_Labels.csv
```

## 3. Train (two-stage fine-tune)

```bash
python -m src.train --data-root data --warmup-epochs 3 --finetune-epochs 15 --batch-size 32
```

Stage 1 trains only the classification head (backbone frozen). Stage 2 unfreezes the
backbone and trains with a 10x smaller LR than the head (discriminative fine-tuning).
Best checkpoint by validation macro-AUC is saved to `checkpoints/best.pt`; the training
history is written to `results/history.json`.

## 4. Evaluate

```bash
python -m src.evaluate --data-root data --split val
```

Writes to `results/`:

- `metrics.json` — macro AUC / AP / F1, Hamming loss, disease-risk AUC/AP
- `per_class.csv` — per-class AUC, AP, precision, recall, F1, support
- `roc_disease_risk.png`, `pr_disease_risk.png`, `per_class_auc.png`

## 5. Predict from the CLI

```bash
python -m src.predict --image path/to/fundus.png --checkpoint checkpoints/best.pt
```

## 6. Launch the Streamlit UI

```bash
streamlit run app.py
```

Tabs:

1. **Predict** — upload a fundus image, see top predictions + Grad-CAM overlay.
2. **Model** — architecture diagram, training recipe, parameter counts.
3. **Dataset** — RFMiD overview and training-set label distribution.
4. **Results** — metrics dashboard with ROC/PR curves and per-class table.
5. **About** — pipeline summary and reproduction instructions.

## Project layout

```
src/
  dataset.py    # RFMiDDataset + transforms + pos_weight + label distribution
  model.py     # ResNet50 + freeze/unfreeze + param groups
  train.py     # two-stage fine-tuning loop
  evaluate.py  # metrics + plots
  predict.py   # inference API (used by CLI + app)
  gradcam.py   # Grad-CAM implementation
app.py         # Streamlit multi-tab UI
```

## Disclaimer

Academic prototype — **not a medical device**. Do not use for clinical decisions.
