import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import DISEASE_COLS, RFMiDDataset, compute_pos_weight
from .model import build_model, count_parameters, param_groups, set_backbone_trainable


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, ys, ps = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc="train" if train else "val", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            ys.append(y.detach().cpu())
            ps.append(torch.sigmoid(logits).detach().cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    try:
        auc = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        auc = float("nan")
    return total_loss / len(loader.dataset), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--warmup-epochs", type=int, default=3,
                    help="epochs with backbone frozen (head only)")
    ap.add_argument("--finetune-epochs", type=int, default=15,
                    help="epochs with full network unfrozen")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"device={device}")

    train_ds = RFMiDDataset(args.data_root, "train")
    val_ds = RFMiDDataset(args.data_root, "val")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(pretrained=True).to(device)
    pos_weight = compute_pos_weight(args.data_root).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    history = []
    best_auc = -1.0

    def save_best(epoch, stage, val_auc):
        torch.save({
            "model": model.state_dict(),
            "labels": DISEASE_COLS,
            "epoch": epoch,
            "stage": stage,
            "val_auc": val_auc,
        }, Path(args.ckpt_dir) / "best.pt")

    # ---------- Stage 1: warm up head ----------
    print("\n== Stage 1: warm-up (backbone frozen) ==")
    set_backbone_trainable(model, False)
    print("params:", count_parameters(model))
    optimizer = torch.optim.AdamW(
        param_groups(model, head_lr=args.head_lr, backbone_lr=args.backbone_lr),
        weight_decay=1e-4,
    )
    for epoch in range(1, args.warmup_epochs + 1):
        tr_loss, tr_auc = run_epoch(model, train_dl, criterion, optimizer, device, True)
        va_loss, va_auc = run_epoch(model, val_dl, criterion, optimizer, device, False)
        history.append({"stage": "warmup", "epoch": epoch,
                        "train_loss": tr_loss, "train_auc": tr_auc,
                        "val_loss": va_loss, "val_auc": va_auc})
        print(f"[warmup {epoch:02d}] tr {tr_loss:.4f}/{tr_auc:.4f} "
              f"va {va_loss:.4f}/{va_auc:.4f}")
        if va_auc > best_auc:
            best_auc = va_auc
            save_best(epoch, "warmup", va_auc)

    # ---------- Stage 2: full fine-tune ----------
    print("\n== Stage 2: full fine-tune (all layers) ==")
    set_backbone_trainable(model, True)
    print("params:", count_parameters(model))
    optimizer = torch.optim.AdamW(
        param_groups(model, head_lr=args.head_lr * 0.5, backbone_lr=args.backbone_lr),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs)
    for epoch in range(1, args.finetune_epochs + 1):
        tr_loss, tr_auc = run_epoch(model, train_dl, criterion, optimizer, device, True)
        va_loss, va_auc = run_epoch(model, val_dl, criterion, optimizer, device, False)
        scheduler.step()
        history.append({"stage": "finetune", "epoch": epoch,
                        "train_loss": tr_loss, "train_auc": tr_auc,
                        "val_loss": va_loss, "val_auc": va_auc})
        print(f"[finetune {epoch:02d}] tr {tr_loss:.4f}/{tr_auc:.4f} "
              f"va {va_loss:.4f}/{va_auc:.4f}")
        if va_auc > best_auc:
            best_auc = va_auc
            save_best(epoch, "finetune", va_auc)

    with open(Path(args.results_dir) / "history.json", "w") as f:
        json.dump({"history": history, "best_val_auc": best_auc}, f, indent=2)
    print(f"\nbest val macro-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
