import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import DISEASE_COLS, DISEASE_NAMES, RFMiDDataset
from .model import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in tqdm(loader, desc="eval"):
        x = x.to(device)
        ps.append(torch.sigmoid(model(x)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def safe_auc(y, p):
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return float("nan")


def per_class_metrics(y_true, y_pred, threshold=0.5):
    rows = []
    for i, code in enumerate(DISEASE_COLS):
        yt, yp = y_true[:, i], y_pred[:, i]
        yb = (yp >= threshold).astype(int)
        rows.append({
            "code": code,
            "name": DISEASE_NAMES[code],
            "support": int(yt.sum()),
            "auc": safe_auc(yt, yp),
            "ap": average_precision_score(yt, yp) if yt.sum() > 0 else float("nan"),
            "precision": precision_score(yt, yb, zero_division=0),
            "recall": recall_score(yt, yb, zero_division=0),
            "f1": f1_score(yt, yb, zero_division=0),
        })
    return pd.DataFrame(rows)


def plot_roc(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y_true[:, 0], y_pred[:, 0])
    ax.plot(fpr, tpr, label=f"Disease_Risk (AUC={safe_auc(y_true[:,0], y_pred[:,0]):.3f})",
            linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC - Overall Disease Risk")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    prec, rec, _ = precision_recall_curve(y_true[:, 0], y_pred[:, 0])
    ap = average_precision_score(y_true[:, 0], y_pred[:, 0])
    ax.plot(rec, prec, label=f"Disease_Risk (AP={ap:.3f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall - Overall Disease Risk")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_auc(df, out_path, top_n=20):
    d = df.dropna(subset=["auc"]).sort_values("auc", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(d))))
    ax.barh(d["code"], d["auc"], color="#4c8bf5")
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC-AUC")
    ax.set_title(f"Per-class AUC (top {len(d)} by support-weighted score)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=False)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)

    ds = RFMiDDataset(args.data_root, args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    y_true, y_pred = collect_predictions(model, dl, device)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = per_class_metrics(y_true, y_pred, threshold=args.threshold)
    df.to_csv(out / "per_class.csv", index=False)

    metrics = {
        "split": args.split,
        "num_samples": int(len(ds)),
        "macro_auc": float(np.nanmean(df["auc"])),
        "macro_ap": float(np.nanmean(df["ap"])),
        "macro_f1": float(df["f1"].mean()),
        "hamming_loss": float(hamming_loss(y_true, (y_pred >= args.threshold).astype(int))),
        "disease_risk_auc": float(safe_auc(y_true[:, 0], y_pred[:, 0])),
        "disease_risk_ap": float(average_precision_score(y_true[:, 0], y_pred[:, 0])),
        "threshold": args.threshold,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc(y_true, y_pred, out / "roc_disease_risk.png")
    plot_pr(y_true, y_pred, out / "pr_disease_risk.png")
    plot_per_class_auc(df, out / "per_class_auc.png")

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote metrics + plots to {out}/")


if __name__ == "__main__":
    main()
