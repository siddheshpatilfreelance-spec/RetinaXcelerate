"""Enhanced mode — paper Future Work §VII-1 ("add a fine-tuned member to the existing
ensemble"). Same CCER aggregation as the paper (Eq. 2), with four members:

  rafalosa, Kontawat : paper members, paper dual-path Φ (Eq. 3) on the circular crop
  vit, effnet        : members fine-tuned by src.train_dr on APTOS 2019 + IDRiD

Weights come from checkpoints/ccer_weights.json (chosen on validation by
src.evaluate_ccer). Explainability stays Attention Rollout (paper §III-D), taken from the
fine-tuned ViT member.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dr_model import DRPredictor, preprocess_pil
from .hf_model import DR_CLASSES, EnsembleDR, circle_crop


class EnhancedCCER:
    def __init__(self, paper_ens: EnsembleDR, ckpt_dir: str | Path = "checkpoints"):
        ckpt_dir = Path(ckpt_dir)
        self.paper = paper_ens
        self.tuned = DRPredictor(ckpt_dir, device=paper_ens.device)
        self.weights = json.loads((ckpt_dir / "ccer_weights.json").read_text())
        self.device = paper_ens.device

    def total_params(self) -> int:
        return self.paper.total_params() + self.tuned.total_params()

    def predict(self, image: Image.Image) -> dict:
        per = {}
        c = circle_crop(image)
        for m in self.paper.members:
            name = m["spec"].repo.split("/")[0]
            if self.weights.get(name, 0) > 0:
                per[name] = self.paper._member_probs(m, c)
        img512 = preprocess_pil(image)
        for m in self.tuned.members:
            if self.weights.get(m["arch"], 0) > 0:
                per[m["arch"]] = self.tuned.member_probs(m, img512)
        wsum = sum(self.weights[k] for k in per)
        probs = sum(self.weights[k] * p for k, p in per.items()) / wsum
        probs = probs / probs.sum()
        idx = int(np.argmax(probs))
        vit = next((m for m in self.tuned.members if m["arch"] == "vit"), None)
        if vit is not None:
            heat, base = self.tuned._rollout(vit, img512), img512
        else:
            heat, _ = self.paper.attention(c)
            base = c
        return {
            "probs": probs, "pred_idx": idx, "pred": DR_CLASSES[idx],
            "ranking": sorted([{**DR_CLASSES[i], "prob": float(probs[i])} for i in range(5)],
                              key=lambda r: r["prob"], reverse=True),
            "per_model": {k: v.tolist() for k, v in per.items()},
            "heatmap": heat, "heat_base": base,
        }
