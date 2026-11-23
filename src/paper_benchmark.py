"""Reproduce the tables of the RetinaXcelerate paper from code.

    python -m src.paper_benchmark            # Tables IV, V, VI  -> results/paper_tables.md
    python -m src.paper_benchmark --runs 50  # latency runs (paper: median over n = 50)

Table IV  — top-1 per gallery sample: Kontawat alone, rafalosa alone, CCER ensemble
Table V   — ablation on the `proliferative` sample (TTA / Ben Graham / crop)
Table VI  — per-stage inference latency, median over N runs
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .hf_model import (attention_heatmap, ben_graham, circle_crop, load_hf_model,
                       overlay_heatmap)

GALLERY = ["normal.jpg", "dr_general.png", "dr_eda03.jpg", "cotton_wool.png",
           "proliferative.jpg", "laser_dr.jpg", "cmv.jpg"]
SHORT = {0: "No DR", 1: "Mild", 2: "Mod.", 3: "Severe", 4: "Prolif."}
ALPHA = 0.65  # raw-vs-Ben-Graham mixing coefficient (paper Eq. 3)


def fmt(p: np.ndarray) -> str:
    i = int(np.argmax(p))
    return f"{SHORT[i]} ({p[i]:.2f})"


def member_probs(ens, member, img, tta: bool, bg: bool) -> np.ndarray:
    def path(x):
        views = [x, ImageOps.mirror(x)] if tta else [x]
        return np.mean([ens._one_forward(member, v) for v in views], axis=0)
    raw = path(img)
    return ALPHA * raw + (1 - ALPHA) * path(ben_graham(img)) if bg else raw


def ensemble(ens, img, tta, bg, crop) -> np.ndarray:
    x = circle_crop(img) if crop else img
    p = np.mean([member_probs(ens, m, x, tta, bg) for m in ens.members], axis=0)
    return p / p.sum()


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    ens, _, dev = load_hf_model()
    by = {m["spec"].repo.split("/")[0]: m for m in ens.members}
    imgs = {g: Image.open(Path("static/samples") / g).convert("RGB") for g in GALLERY}
    out = ["## Table IV — top-1 per sample", "",
           "| Sample | Kontawat | rafalosa | Ensemble |", "|---|---|---|---|"]
    for g, img in imgs.items():
        # baselines = each checkpoint alone on the raw image (single view)
        k = member_probs(ens, by["Kontawat"], img, tta=False, bg=False)
        r = member_probs(ens, by["rafalosa"], img, tta=False, bg=False)
        e = ensemble(ens, img, tta=True, bg=True, crop=True)
        out.append(f"| {Path(g).stem} | {fmt(k)} | {fmt(r)} | **{fmt(e)}** |")

    img = imgs["proliferative.jpg"]
    out += ["", "## Table V — ablation on `proliferative`", "",
            "| Configuration | Top-1 | Prob. | Disease? |", "|---|---|---|---|"]
    def bg_only(x):  # "Ensemble + BG": Ben Graham path only, single view
        p = np.mean([ens._one_forward(m, ben_graham(x)) for m in ens.members], axis=0)
        return p / p.sum()
    rows = [("Kontawat only (baseline)", member_probs(ens, by["Kontawat"], img, False, False)),
            ("rafalosa only (baseline)", member_probs(ens, by["rafalosa"], img, False, False)),
            ("Ensemble, no TTA, no BG", ensemble(ens, img, False, False, False)),
            ("Ensemble + TTA", ensemble(ens, img, True, False, False)),
            ("Ensemble + BG", bg_only(img)),
            ("Ens. + TTA + BG + crop", ensemble(ens, img, True, True, True))]
    for name, p in rows:
        i = int(np.argmax(p))
        out.append(f"| {name} | {SHORT[i]} | {p[i]:.2f} | {'yes' if i >= 2 else ('partial' if i == 1 else 'no')} |")

    # ---- Table VI: per-stage latency (end-to-end request path of /api/predict) ----
    stages = {k: [] for k in ["Circular crop", "Ben Graham transform",
                              "Member 1 forward (raw + h-flip)", "Member 1 forward (BG + h-flip)",
                              "Member 2 forward (raw + h-flip)", "Member 2 forward (BG + h-flip)",
                              "CCER aggregation", "Attention Rollout", "Overlay + base64 encode"]}
    totals = []
    for run in range(args.runs + 2):          # 2 warm-up runs are discarded
        t = {}
        T0 = time.perf_counter()
        s = time.perf_counter(); x = circle_crop(img); t["Circular crop"] = time.perf_counter() - s
        s = time.perf_counter(); xb = ben_graham(x); t["Ben Graham transform"] = time.perf_counter() - s
        per = []
        for i, m in enumerate(ens.members, 1):
            s = time.perf_counter()
            pr = np.mean([ens._one_forward(m, v) for v in (x, ImageOps.mirror(x))], 0); sync(dev)
            t[f"Member {i} forward (raw + h-flip)"] = time.perf_counter() - s
            s = time.perf_counter()
            pg = np.mean([ens._one_forward(m, v) for v in (xb, ImageOps.mirror(xb))], 0); sync(dev)
            t[f"Member {i} forward (BG + h-flip)"] = time.perf_counter() - s
            per.append(ALPHA * pr + (1 - ALPHA) * pg)
        s = time.perf_counter(); p = np.mean(per, 0); p /= p.sum(); t["CCER aggregation"] = time.perf_counter() - s
        s = time.perf_counter(); heat, hw = ens.attention(x); sync(dev); t["Attention Rollout"] = time.perf_counter() - s
        s = time.perf_counter()
        buf = io.BytesIO(); overlay_heatmap(img, attention_heatmap(("rollout", heat), hw)).resize((512, 512)).save(buf, "PNG")
        base64.b64encode(buf.getvalue()); t["Overlay + base64 encode"] = time.perf_counter() - s
        if run >= 2:
            for k, v in t.items():
                stages[k].append(v)
            totals.append(time.perf_counter() - T0)
    med = {k: statistics.median(v) for k, v in stages.items()}
    tot = statistics.median(totals)
    out += ["", f"## Table VI — per-stage latency on {dev.upper()}, median over n = {args.runs}", "",
            "| Stage | Time (s) | Share |", "|---|---|---|"]
    out += [f"| {k} | {v:.3f} | {100 * v / tot:.0f}% |" for k, v in med.items()]
    out.append(f"| **Total** | **{tot:.2f}** | 100% |")

    Path("results").mkdir(exist_ok=True)
    Path("results/paper_tables.md").write_text("\n".join(out) + "\n")
    json.dump({"latency_median_s": med, "total_median_s": tot, "device": dev, "runs": args.runs},
              open("results/paper_latency.json", "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
