"""Enhanced mode = paper Future Work §VII-1: add fine-tuned members to the CCER ensemble.

Members:
  * rafalosa, Kontawat  — the paper's two HF ViT-B/16 checkpoints, using the paper's
                          dual-path inference Φ(m, x) = 0.65·TTA(x) + 0.35·TTA(BG(x))  (Eq. 3)
  * vit, effnet         — checkpoints fine-tuned by src.train_dr (flip TTA)
CCER weights w_m (Eq. 2) are grid-searched on the *validation* split only, then the
result is evaluated once on the held-out IDRiD test and Messidor-2 sets.

    python -m src.evaluate_ccer
Writes checkpoints/ccer_weights.json, results/ccer_eval.json, results/ccer_probs_<split>.npz
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .dr_model import ARCHS, build_transforms, default_device, load_checkpoint
from .evaluate_dr import metrics
from .hf_model import EnsembleDR
from .train_dr import DRDataset, predict_probs

SPLITS = "data/dr/splits_hires.csv"
IMG_DIR = Path("data/dr/images_hires")
MEMBERS = ["rafalosa", "Kontawat", "vit", "effnet"]
GRID = [0.0, 0.5, 1.0, 1.5, 2.0]


def member_probs(df: pd.DataFrame, split: str) -> dict:
    cache = Path(f"results/ccer_probs_{split}.npz")
    if cache.exists():
        return dict(np.load(cache))
    device = default_device()
    sub = df[df.split == split]
    out = {}
    # fine-tuned members
    for arch in ARCHS:
        model, _ = load_checkpoint(Path("checkpoints") / f"dr_{arch}.pt", device)
        dl = DataLoader(DRDataset(sub, IMG_DIR, build_transforms(arch, False)),
                        batch_size=32, num_workers=4)
        out[arch] = predict_probs(model, dl, device)
        del model
    # paper members (exact paper preprocessing path, Eq. 3)
    ens = EnsembleDR(device)
    hf = {m["spec"].repo.split("/")[0]: [] for m in ens.members}
    for k, iid in enumerate(sub.id):
        img = Image.open(IMG_DIR / f"{iid}.jpg").convert("RGB")
        from .hf_model import circle_crop
        c = circle_crop(img)
        for m in ens.members:
            hf[m["spec"].repo.split("/")[0]].append(ens._member_probs(m, c))
        if (k + 1) % 200 == 0:
            print(f"  {split}: {k + 1}/{len(sub)}", flush=True)
    out.update({k: np.array(v) for k, v in hf.items()})
    del ens
    if device == "mps":
        torch.mps.empty_cache()
    np.savez(cache, **out)
    return out


def combine(P: dict, w: dict) -> np.ndarray:
    tot = sum(w.values())
    p = sum(w[m] * P[m] for m in MEMBERS) / tot
    return p / p.sum(1, keepdims=True)


def main():
    df = pd.read_csv(SPLITS)
    P = {s: member_probs(df, s) for s in ["val", "test", "external"]}
    yv = df[df.split == "val"].label.to_numpy()

    best = None
    for ws in itertools.product(GRID, repeat=len(MEMBERS)):
        if sum(ws) == 0:
            continue
        w = dict(zip(MEMBERS, ws))
        q = metrics(yv, combine(P["val"], w))["qwk"]
        if best is None or q > best[0] + 1e-9:
            best = (q, w)
    q, w = best
    print(f"chosen CCER weights (val QWK {q:.3f}): {w}")
    json.dump(w, open("checkpoints/ccer_weights.json", "w"), indent=1)

    res = {"weights": w, "val_qwk": q}
    configs = {
        "Paper CCER (rafalosa + Kontawat, equal weights)": {"rafalosa": 1, "Kontawat": 1, "vit": 0, "effnet": 0},
        "Fine-tuned only (ViT + EfficientNet, val-tuned)": {"rafalosa": 0, "Kontawat": 0,
                                                            **json.load(open("checkpoints/ensemble_weights.json"))},
        "Enhanced CCER (4 members, val-tuned)": w,
    }
    for s in ["test", "external"]:
        y = df[df.split == s].label.to_numpy()
        res[s] = {name: metrics(y, combine(P[s], cw)) for name, cw in configs.items()}
        print(f"\n== {s} (n={len(y)})")
        for name, m in res[s].items():
            print(f"  {name:52s} acc {m['accuracy']:.3f}  QWK {m['qwk']:.3f}  "
                  f"refSens {m['referable_sensitivity']:.3f}  refSpec {m['referable_specificity']:.3f}")
    json.dump(res, open("results/ccer_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
