"""Fine-tune a 5-grade DR classifier on the preprocessed APTOS + IDRiD data.

    python -m src.train_dr --arch vit    --epochs 10
    python -m src.train_dr --arch effnet --epochs 14 --img-dir data/dr/images_hires

Model selection uses the validation split's quadratic weighted kappa (QWK).
The IDRiD test split is never touched here — see src.evaluate_dr.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, Dataset

from .dr_model import ARCHS, build_model, build_transforms, default_device


class DRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, tf):
        self.ids, self.labels = df["id"].tolist(), df["label"].to_numpy()
        self.img_dir, self.tf = Path(img_dir), tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / f"{self.ids[i]}.jpg").convert("RGB")
        return self.tf(img), int(self.labels[i])


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        p = torch.softmax(model(x), -1) + torch.softmax(model(torch.flip(x, dims=[3])), -1)
        out.append((p / 2).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="backbone LR (head gets 10x)")
    ap.add_argument("--splits", default="data/dr/splits.csv")
    ap.add_argument("--img-dir", default="data/dr/images")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = default_device()
    lr = args.lr or {"vit": 3e-5, "effnet": 2e-4}[args.arch]

    df = pd.read_csv(args.splits)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    counts = np.bincount(tr.label, minlength=5)
    # sqrt-inverse-frequency class weights: helps the rare Mild / Severe grades
    cw = (counts.sum() / counts) ** 0.5
    cw = torch.tensor(cw / cw.mean(), dtype=torch.float32)
    print(f"arch={args.arch} device={device} train={len(tr)} val={len(va)} "
          f"class_weights={np.round(cw.numpy(), 2).tolist()}", flush=True)

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                 persistent_workers=args.workers > 0)
    train_dl = DataLoader(DRDataset(tr, args.img_dir, build_transforms(args.arch, True)),
                          shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(DRDataset(va, args.img_dir, build_transforms(args.arch, False)),
                        shuffle=False, **dl_kw)

    model = build_model(args.arch).to(device)
    head = {id(p) for p in model.head_params()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head], "lr": lr},
        {"params": model.head_params(), "lr": lr * 10},
    ], weight_decay=0.05)
    steps = args.epochs * len(train_dl)
    warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    loss_fn = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=0.05)

    Path(args.out).mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    best, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        probs = predict_probs(model, val_dl, device)
        pred, yv = probs.argmax(1), va.label.to_numpy()
        src = va.source.to_numpy()
        rec = {"epoch": ep, "train_loss": tot / n, "val_acc": accuracy_score(yv, pred),
               "val_qwk": cohen_kappa_score(yv, pred, weights="quadratic"),
               "val_acc_idrid": accuracy_score(yv[src == "idrid"], pred[src == "idrid"]),
               "val_acc_aptos": accuracy_score(yv[src == "aptos"], pred[src == "aptos"]),
               "minutes": (time.time() - t0) / 60}
        history.append(rec)
        flag = ""
        if rec["val_qwk"] > best:
            best = rec["val_qwk"]
            flag = "  * saved"
            torch.save({"arch": args.arch, "state_dict": model.state_dict(),
                        "epoch": ep, "val": rec, "res": ARCHS[args.arch]["res"],
                        "img_dir": args.img_dir}, Path(args.out) / f"dr_{args.arch}.pt")
        print(f"ep {ep:2d} loss {rec['train_loss']:.3f} | val acc {rec['val_acc']:.3f} "
              f"QWK {rec['val_qwk']:.3f} (idrid {rec['val_acc_idrid']:.3f}, "
              f"aptos {rec['val_acc_aptos']:.3f}) | {rec['minutes']:.1f} min{flag}", flush=True)
        json.dump(history, open(f"results/dr_{args.arch}_history.json", "w"), indent=1)
    print(f"best val QWK {best:.3f}")


if __name__ == "__main__":
    main()
