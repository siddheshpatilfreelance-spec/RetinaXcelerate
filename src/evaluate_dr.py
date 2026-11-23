"""Evaluate trained DR checkpoints on held-out data (never used for training/selection).

    python -m src.evaluate_dr --splits data/dr/splits_hires.csv --img-dir data/dr/images_hires

Evaluates every checkpoints/dr_*.pt alone and the probability-averaged ensemble on:
  * test      — IDRiD official test split (103 images, ophthalmologist-graded)
  * external  — Messidor-2 (1,744 images, different country/camera; if prepared)
The ensemble weight is chosen on the *validation* split only (grid search on QWK)
and saved to checkpoints/ensemble_weights.json, which the web app uses.
Writes results/dr_eval.json and results/dr_eval.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score)
from torch.utils.data import DataLoader

from .dr_model import ARCHS, build_transforms, default_device, load_checkpoint
from .train_dr import DRDataset, predict_probs

NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


def metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    ref_t, ref_p = y >= 2, pred >= 2  # referable DR = moderate or worse
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(5), zero_division=0)),
        "recall_per_class": recall_score(y, pred, average=None, labels=range(5),
                                         zero_division=0).round(3).tolist(),
        "referable_sensitivity": float((ref_p & ref_t).sum() / max(ref_t.sum(), 1)),
        "referable_specificity": float((~ref_p & ~ref_t).sum() / max((~ref_t).sum(), 1)),
        "confusion": confusion_matrix(y, pred, labels=range(5)).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/dr/splits_hires.csv")
    ap.add_argument("--img-dir", default="data/dr/images_hires")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    device = default_device()
    df = pd.read_csv(args.splits)
    evals = [s for s in ["val", "test", "external"] if (df.split == s).any()]
    ckpts = [a for a in ARCHS if (Path(args.ckpt_dir) / f"dr_{a}.pt").exists()]
    probs = {s: {} for s in evals}
    for arch in ckpts:
        model, ck = load_checkpoint(Path(args.ckpt_dir) / f"dr_{arch}.pt", device)
        print(f"{arch}: epoch {ck['epoch']}, val QWK {ck['val']['val_qwk']:.3f}", flush=True)
        for s in evals:
            sub = df[df.split == s]
            dl = DataLoader(DRDataset(sub, args.img_dir, build_transforms(arch, False)),
                            batch_size=32, num_workers=4)
            probs[s][arch] = predict_probs(model, dl, device)
        del model
        torch.mps.empty_cache() if device == "mps" else None

    # ---- choose ensemble weights on validation only ----
    weights = {a: 1.0 for a in ckpts}
    if len(ckpts) == 2:
        yv = df[df.split == "val"].label.to_numpy()
        a0, a1 = ckpts
        grid = []
        for w in np.round(np.arange(0, 1.01, 0.1), 2):
            pv = (1 - w) * probs["val"][a0] + w * probs["val"][a1]
            grid.append((cohen_kappa_score(yv, pv.argmax(1), weights="quadratic"), w))
        best_qwk, w = max(grid)
        weights = {a0: float(1 - w), a1: float(w)}
        print("val QWK by weight on", a1, ":", [(w_, round(q, 3)) for q, w_ in grid])
        print("chosen weights:", weights, "val QWK", round(best_qwk, 3))
    json.dump(weights, open(Path(args.ckpt_dir) / "ensemble_weights.json", "w"), indent=1)

    results = {"ensemble_weights": weights}
    evals = [s for s in evals if s != "val"]
    for s in evals:
        y = df[df.split == s].label.to_numpy()
        results[s] = {"label_counts": np.bincount(y, minlength=5).tolist()}
        for arch in ckpts:
            results[s][ARCHS[arch]["label"]] = metrics(y, probs[s][arch])
        if len(ckpts) > 1:
            results[s]["Ensemble (equal weights)"] = metrics(y, np.mean(list(probs[s].values()), 0))
            pw = sum(weights[a] * probs[s][a] for a in ckpts) / sum(weights.values())
            results[s]["Ensemble (val-tuned weights)"] = metrics(y, pw)
        np.save(f"results/dr_probs_{s}.npy", np.stack([probs[s][a] for a in ckpts]))

    Path("results").mkdir(exist_ok=True)
    json.dump(results, open("results/dr_eval.json", "w"), indent=1)
    lines = [f"Ensemble weights (chosen on validation): {weights}", ""]
    for s in evals:
        title = {"test": "IDRiD test", "external": "Messidor-2 (external)"}[s]
        lines += [f"### {title} — n={sum(results[s]['label_counts'])}, "
                  f"labels {results[s]['label_counts']}", "",
                  "| Model | Acc | QWK | Macro-F1 | Recall No/Mild/Mod/Sev/PDR | Ref. sens | Ref. spec |",
                  "|---|---|---|---|---|---|---|"]
        for k, m in results[s].items():
            if k in ("label_counts",):
                continue
            lines.append(f"| {k} | {m['accuracy']:.3f} | {m['qwk']:.3f} | {m['macro_f1']:.3f} | "
                         f"{' / '.join(f'{r:.2f}' for r in m['recall_per_class'])} | "
                         f"{m['referable_sensitivity']:.3f} | {m['referable_specificity']:.3f} |")
        lines.append("")
    open("results/dr_eval.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
