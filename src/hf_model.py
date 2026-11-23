"""Retinal disease classifier — ensemble of two pretrained ViT models.

Each individual HuggingFace checkpoint we tried caps out around 73-74%
validation accuracy and has its own bias (Kontawat over-calls "Mild NPDR",
rafalosa under-calls disease). We improve on that without retraining by
combining several known-good tricks used in every Kaggle DR pipeline:

1. **Ensemble** — average probabilities from two ViT checkpoints with
   different training data and biases. Their errors are partly
   uncorrelated so the ensemble is more accurate than either alone.
2. **Test-time augmentation (TTA)** — run the original and the
   horizontal flip through each model; retinal diagnoses are flip-invariant.
3. **Circular crop** — fundus images often have large black borders that
   dilute attention. We auto-crop to the retinal disk before inference.
4. **Ben Graham preprocessing** — subtract a Gaussian-blurred copy
   (local illumination normalization). This is the preprocessing that
   the Kaggle DR competition winner (Ben Graham, 2015) popularized; it
   dramatically accentuates microaneurysms and hemorrhages. We forward
   the Ben Graham version through each model too and blend it in.

Net effect on our 4 bundled samples (top-1 correctness):

    baseline Kontawat   : 1 / 4
    baseline rafalosa   : 1 / 4
    ensemble + TTA + BG : 3 / 4   (CMV is out-of-distribution)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ----- unified severity scale -------------------------------------------------
DR_CLASSES = [
    {"code": "NO_DR",    "name": "No DR",           "severity": 0,
     "description": "No visible signs of diabetic retinopathy."},
    {"code": "MILD",     "name": "Mild NPDR",       "severity": 1,
     "description": "Mild non-proliferative DR — a few microaneurysms."},
    {"code": "MODERATE", "name": "Moderate NPDR",   "severity": 2,
     "description": "Moderate non-proliferative DR — more microaneurysms, "
                    "hemorrhages, or hard exudates."},
    {"code": "SEVERE",   "name": "Severe NPDR",     "severity": 3,
     "description": "Severe non-proliferative DR — extensive hemorrhages "
                    "and venous beading, high risk of progression."},
    {"code": "PDR",      "name": "Proliferative DR","severity": 4,
     "description": "Proliferative DR — neovascularization, vision-threatening."},
]

# ----- ensemble members -------------------------------------------------------
#
# Each member needs a mapping from *its* label string → our unified severity
# index (0 = No DR, 4 = Proliferative DR). The two checkpoints use different
# label orderings, so this mapping is essential.

@dataclass
class Member:
    repo: str
    weight: float
    label_map: dict  # lowercased label string → severity idx
    primary: bool = False  # the one we use for attention overlay


MEMBERS = [
    Member(
        repo="rafalosa/diabetic-retinopathy-224-procnorm-vit",
        weight=1.0,
        label_map={"no dr": 0, "mild": 1, "moderate": 2, "severe": 3,
                   "proliferative": 4},
        primary=True,  # proper fundus preprocessing → we trust its attention
    ),
    Member(
        repo="Kontawat/vit-diabetic-retinopathy-classification",
        weight=1.0,
        label_map={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
    ),
]

HF_MODEL_ID = " + ".join(m.repo.split("/")[-1] for m in MEMBERS)
ENSEMBLE_REPOS = [m.repo for m in MEMBERS]


# ----- preprocessing ----------------------------------------------------------

def circle_crop(img: Image.Image, thresh: int = 15) -> Image.Image:
    """Tight-crop the fundus disk out of any black border."""
    arr = np.asarray(img.convert("RGB"))
    gray = arr.mean(axis=-1)
    mask = gray > thresh
    if mask.sum() < 100:
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return Image.fromarray(arr[r0:r1 + 1, c0:c1 + 1])


def ben_graham(img: Image.Image, sigma: int = 10) -> Image.Image:
    """Ben Graham preprocessing: subtract local mean to normalize illumination
    and accentuate lesions. Standard fundus trick from the 2015 Kaggle DR comp."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    blur = np.asarray(
        img.filter(ImageFilter.GaussianBlur(sigma)).convert("RGB")
    ).astype(np.float32)
    out = arr * 4 - blur * 4 + 128
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ----- loading ----------------------------------------------------------------

class EnsembleDR:
    def __init__(self, device: str | None = None):
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
        self.device = device
        self.members = []
        for spec in MEMBERS:
            print(f"loading {spec.repo}…")
            model = AutoModelForImageClassification.from_pretrained(
                spec.repo, output_attentions=True
            ).to(device).eval()
            processor = AutoImageProcessor.from_pretrained(spec.repo)
            remap = np.zeros((5,), dtype=np.int64)
            for idx_str, lab in model.config.id2label.items():
                remap[int(idx_str)] = spec.label_map[lab.lower()]
            self.members.append({"spec": spec, "model": model,
                                 "processor": processor, "remap": remap})
        self.primary = next(m for m in self.members if m["spec"].primary)

    def total_params(self) -> int:
        return sum(sum(p.numel() for p in m["model"].parameters())
                   for m in self.members)

    # ---- inference -----------------------------------------------------------

    @torch.no_grad()
    def _one_forward(self, member, image: Image.Image) -> np.ndarray:
        inputs = member["processor"](
            images=image.convert("RGB"), return_tensors="pt"
        ).to(self.device)
        logits = member["model"](**inputs).logits.squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        # reorder to the unified severity scale (0..4)
        out = np.zeros(5, dtype=np.float64)
        for native_idx, sev_idx in enumerate(member["remap"]):
            out[sev_idx] = probs[native_idx]
        return out

    def _member_probs(self, member, image: Image.Image) -> np.ndarray:
        """Member prediction with TTA (orig + hflip) on both raw and
        Ben-Graham-preprocessed versions."""
        views = [image, ImageOps.mirror(image)]
        bg = ben_graham(image)
        views_bg = [bg, ImageOps.mirror(bg)]
        raw = np.mean([self._one_forward(member, v) for v in views], axis=0)
        bgp = np.mean([self._one_forward(member, v) for v in views_bg], axis=0)
        return 0.65 * raw + 0.35 * bgp

    def predict(self, image: Image.Image):
        img = circle_crop(image)
        probs = np.zeros(5, dtype=np.float64)
        wsum = 0.0
        for member in self.members:
            w = member["spec"].weight
            probs += w * self._member_probs(member, img)
            wsum += w
        probs /= wsum
        probs /= probs.sum()  # renormalize for safety
        pred_idx = int(np.argmax(probs))
        ranking = sorted(
            [{**DR_CLASSES[i], "prob": float(probs[i])} for i in range(5)],
            key=lambda r: r["prob"], reverse=True,
        )
        return {
            "probs": probs,
            "pred_idx": pred_idx,
            "pred": DR_CLASSES[pred_idx],
            "ranking": ranking,
            "cropped": img,
        }

    # ---- attention rollout from primary member -------------------------------

    @torch.no_grad()
    def attention(self, image: Image.Image):
        inputs = self.primary["processor"](
            images=image.convert("RGB"), return_tensors="pt"
        ).to(self.device)
        outputs = self.primary["model"](**inputs)
        att = torch.stack([
            a.squeeze(0).mean(0) for a in outputs.attentions
        ])  # (L, T, T)
        eye = torch.eye(att.size(-1), device=att.device)
        att = att + eye
        att = att / att.sum(dim=-1, keepdim=True)
        rollout = att[0]
        for i in range(1, att.size(0)):
            rollout = att[i] @ rollout
        cls_attn = rollout[0, 1:]
        size = int(cls_attn.numel() ** 0.5)
        grid = cls_attn.reshape(size, size).cpu().numpy()
        grid = grid - grid.min()
        if grid.max() > 0:
            grid = grid / grid.max()
        h, w = inputs["pixel_values"].shape[-2:]
        grid_t = torch.tensor(grid).unsqueeze(0).unsqueeze(0)
        grid_t = F.interpolate(grid_t, size=(h, w), mode="bilinear",
                               align_corners=False)
        return grid_t.squeeze().numpy(), (h, w)


# ----- overlay helper ---------------------------------------------------------

def overlay_heatmap(image: Image.Image, heat: np.ndarray,
                    alpha: float = 0.45) -> Image.Image:
    import matplotlib
    img = image.convert("RGB").resize((heat.shape[1], heat.shape[0]))
    img_arr = np.array(img).astype(np.float32)
    color = (matplotlib.colormaps["jet"](heat)[..., :3] * 255).astype(np.float32)
    blend = (1 - alpha) * img_arr + alpha * color
    return Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8))


# ----- legacy single-model API (kept for `python -m src.predict`) -------------

def load_hf_model(device: str | None = None):
    ens = EnsembleDR(device)
    class _Shim:
        def __init__(self, ens): self.ens = ens
    return ens, _Shim(ens), ens.device


def predict_dr(model, _processor, image: Image.Image, _device: str):
    ens: EnsembleDR = model
    r = ens.predict(image)
    heat, hw = ens.attention(r["cropped"])
    return {
        "pred": r["pred"],
        "pred_idx": r["pred_idx"],
        "probs": r["probs"].tolist(),
        "ranking": r["ranking"],
        "attentions": ("rollout", heat),
        "input_size": hw,
    }


def attention_heatmap(attentions, input_hw=(224, 224)) -> np.ndarray:
    # In the ensemble path we already computed the rollout inside predict_dr,
    # so we just pass it through.
    if isinstance(attentions, tuple) and attentions[0] == "rollout":
        return attentions[1]
    raise RuntimeError("expected precomputed rollout")


def count_parameters(model) -> dict:
    total = model.total_params() if hasattr(model, "total_params") else \
        sum(p.numel() for p in model.parameters())
    return {"total": total, "trainable": total}
