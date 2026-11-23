import torch.nn as nn
from torchvision import models

from .dataset import DISEASE_COLS

MODEL_NAME = "ResNet50 (ImageNet pretrained)"
MODEL_DESCRIPTION = """
**Backbone:** ResNet50, pretrained on ImageNet-1k (V2 weights).
**Input:** 384x384 RGB fundus image, normalized with ImageNet statistics.
**Head:** single fully-connected layer producing 46 logits
(1 for overall `Disease_Risk` + 45 fine-grained disease labels).
**Loss:** BCEWithLogitsLoss with per-class `pos_weight` computed from the training
set to counter heavy class imbalance.
**Optimization:** AdamW + cosine LR schedule.
**Fine-tuning schedule (two stage):**
1. *Warm-up (head only)* — backbone frozen, only the new FC layer trains for a few
   epochs so the random head does not destabilize pretrained features.
2. *Full fine-tune* — all layers unfrozen, backbone trained with a 10x smaller LR
   than the head (discriminative learning rates).
""".strip()


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(DISEASE_COLS)),
    )
    return net


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable


def param_groups(model: nn.Module, head_lr: float, backbone_lr: float):
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith("fc.") else backbone).append(p)
    groups = [{"params": head, "lr": head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return groups


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
