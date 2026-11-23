"""RetinaXcelerate DR models: builders, transforms and the inference predictor.

Two transfer-learned backbones, fine-tuned by `src.train_dr` on APTOS 2019 + IDRiD:
  * vit     — ViT-B/16 @224, initialised from a DR-pretrained HuggingFace checkpoint
  * effnet  — EfficientNet-B3 @384, initialised from ImageNet
The predictor averages their probabilities (with horizontal-flip TTA) and produces
an explanation heatmap (Grad-CAM for the CNN, attention rollout for the ViT).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from .prepare_dr_data import preprocess_bytes  # noqa: F401  (re-exported)
from .prepare_dr_data import circle_crop, pad_square

DR_CLASSES = [
    {"code": "NO_DR", "name": "No DR", "severity": 0,
     "description": "No visible signs of diabetic retinopathy."},
    {"code": "MILD", "name": "Mild NPDR", "severity": 1,
     "description": "Mild non-proliferative DR — a few microaneurysms."},
    {"code": "MODERATE", "name": "Moderate NPDR", "severity": 2,
     "description": "Moderate non-proliferative DR — more microaneurysms, "
                    "hemorrhages, or hard exudates."},
    {"code": "SEVERE", "name": "Severe NPDR", "severity": 3,
     "description": "Severe non-proliferative DR — extensive hemorrhages "
                    "and venous beading, high risk of progression."},
    {"code": "PDR", "name": "Proliferative DR", "severity": 4,
     "description": "Proliferative DR — neovascularization, vision-threatening."},
]

VIT_INIT = "Kontawat/vit-diabetic-retinopathy-classification"
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

ARCHS = {
    "vit": {"res": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
            "label": "ViT-B/16 @224"},
    "effnet": {"res": 384, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
               "label": "EfficientNet-B3 @384"},
}


class ViTClassifier(nn.Module):
    """Thin wrapper so a HF ViT behaves like a torchvision model (returns logits)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from transformers import ViTConfig, ViTForImageClassification
        if pretrained:
            self.m = ViTForImageClassification.from_pretrained(
                VIT_INIT, attn_implementation="eager")
        else:
            cfg = ViTConfig.from_pretrained(VIT_INIT, attn_implementation="eager")
            self.m = ViTForImageClassification(cfg)
        # the init checkpoint already uses labels "0".."4" == our severity order

    def forward(self, x, output_attentions: bool = False):
        out = self.m(pixel_values=x, output_attentions=output_attentions)
        return (out.logits, out.attentions) if output_attentions else out.logits

    def head_params(self):
        return list(self.m.classifier.parameters())


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch == "vit":
        return ViTClassifier(pretrained)
    if arch == "effnet":
        from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
        net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        net.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.classifier[1].in_features, 5))
        net.head_params = lambda: list(net.classifier.parameters())
        return net
    raise ValueError(arch)


def build_transforms(arch: str, train: bool):
    a = ARCHS[arch]
    norm = [T.ToTensor(), T.Normalize(a["mean"], a["std"])]
    if not train:
        return T.Compose([T.Resize((a["res"], a["res"]), antialias=True), *norm])
    return T.Compose([
        T.RandomResizedCrop(a["res"], scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        *norm,
    ])


def preprocess_pil(img: Image.Image, size: int = 512) -> Image.Image:
    """Same per-image pipeline as training: circular crop -> pad square -> resize."""
    img = img.convert("RGB")
    if max(img.size) > size * 3:
        img = img.copy()
        img.thumbnail((size * 3, size * 3))
    arr = pad_square(circle_crop(np.asarray(img)))
    return Image.fromarray(arr).resize((size, size), Image.BICUBIC)


def default_device() -> str:
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")


def load_checkpoint(path: str | Path, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


# ----- inference ---------------------------------------------------------------

class DRPredictor:
    def __init__(self, ckpt_dir: str | Path = "checkpoints", device: str | None = None,
                 weights: dict | None = None):
        self.device = device or default_device()
        wfile = Path(ckpt_dir) / "ensemble_weights.json"
        if weights is None and wfile.exists():
            import json
            weights = json.loads(wfile.read_text())  # chosen on validation by src.evaluate_dr
        self.members = []
        for arch in ARCHS:
            p = Path(ckpt_dir) / f"dr_{arch}.pt"
            if p.exists():
                model, ck = load_checkpoint(p, self.device)
                self.members.append({"arch": arch, "model": model, "meta": ck,
                                     "tf": build_transforms(arch, train=False),
                                     "w": (weights or {}).get(arch, 1.0)})
        if not self.members:
            raise FileNotFoundError(f"no dr_*.pt checkpoints in {ckpt_dir}")

    @property
    def description(self) -> str:
        return " + ".join(ARCHS[m["arch"]]["label"] for m in self.members)

    def total_params(self) -> int:
        return sum(sum(p.numel() for p in m["model"].parameters()) for m in self.members)

    @torch.no_grad()
    def member_probs(self, member, img512: Image.Image) -> np.ndarray:
        x = member["tf"](img512).unsqueeze(0).to(self.device)
        x = torch.cat([x, torch.flip(x, dims=[3])])  # TTA: original + horizontal flip
        return F.softmax(member["model"](x), dim=-1).mean(0).cpu().numpy()

    def predict(self, image: Image.Image, explain: bool = True) -> dict:
        img512 = preprocess_pil(image)
        per = {m["arch"]: self.member_probs(m, img512) for m in self.members}
        wsum = sum(m["w"] for m in self.members)
        probs = sum(m["w"] * per[m["arch"]] for m in self.members) / wsum
        idx = int(np.argmax(probs))
        out = {
            "probs": probs, "pred_idx": idx, "pred": DR_CLASSES[idx],
            "per_model": {k: v.tolist() for k, v in per.items()},
            "ranking": sorted([{**DR_CLASSES[i], "prob": float(probs[i])} for i in range(5)],
                              key=lambda r: r["prob"], reverse=True),
            "preprocessed": img512,
        }
        if explain:
            out["heatmap"] = self.explain(img512, idx)
        return out

    def explain(self, img512: Image.Image, class_idx: int) -> np.ndarray:
        by_arch = {m["arch"]: m for m in self.members}
        if "effnet" in by_arch:
            return self._gradcam(by_arch["effnet"], img512, class_idx)
        return self._rollout(by_arch["vit"], img512)

    def _gradcam(self, member, img512, class_idx) -> np.ndarray:
        model = member["model"]
        layer = model.features[-1]
        store = {}
        h1 = layer.register_forward_hook(lambda _m, _i, o: store.__setitem__("a", o))
        h2 = layer.register_full_backward_hook(lambda _m, _gi, go: store.__setitem__("g", go[0]))
        try:
            x = member["tf"](img512).unsqueeze(0).to(self.device).requires_grad_(True)
            model.zero_grad()
            model(x)[0, class_idx].backward()
            w = store["g"].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * store["a"]).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=img512.size[::-1], mode="bilinear", align_corners=False)
        finally:
            h1.remove()
            h2.remove()
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        return cam / cam.max() if cam.max() > 0 else cam

    @torch.no_grad()
    def _rollout(self, member, img512) -> np.ndarray:
        x = member["tf"](img512).unsqueeze(0).to(self.device)
        _, atts = member["model"](x, output_attentions=True)
        att = torch.stack([a.squeeze(0).mean(0) for a in atts])
        att = att + torch.eye(att.size(-1), device=att.device)
        att = att / att.sum(-1, keepdim=True)
        roll = att[0]
        for i in range(1, att.size(0)):
            roll = att[i] @ roll
        cls = roll[0, 1:]
        s = int(cls.numel() ** 0.5)
        g = cls.reshape(1, 1, s, s)
        g = F.interpolate(g, size=img512.size[::-1], mode="bilinear", align_corners=False)
        g = g.squeeze().cpu().numpy()
        g -= g.min()
        return g / g.max() if g.max() > 0 else g


def overlay_heatmap(image: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib
    img = np.asarray(image.convert("RGB").resize((heat.shape[1], heat.shape[0]))).astype(np.float32)
    color = matplotlib.colormaps["jet"](heat)[..., :3] * 255
    return Image.fromarray(np.clip((1 - alpha) * img + alpha * color, 0, 255).astype(np.uint8))
