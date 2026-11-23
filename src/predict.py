import argparse

import torch
from PIL import Image

from .dataset import DISEASE_COLS, DISEASE_NAMES, build_transforms
from .gradcam import GradCAM, overlay_cam, prepare_tensor
from .model import build_model


def load_model(checkpoint: str, device: str):
    model = build_model(pretrained=False)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    labels = state.get("labels", DISEASE_COLS)
    return model, labels


@torch.no_grad()
def _probs(model, x):
    return torch.sigmoid(model(x)).squeeze(0).cpu().tolist()


def predict_image(model, labels, image: Image.Image, device: str,
                  top_k: int = 5, with_cam: bool = False):
    x = prepare_tensor(image, device)
    probs = _probs(model, x)
    scored = sorted(
        [(c, DISEASE_NAMES.get(c, c), p) for c, p in zip(labels, probs)],
        key=lambda t: t[2], reverse=True,
    )
    out = {
        "disease_risk": probs[labels.index("Disease_Risk")],
        "top": scored[:top_k],
        "all": {c: p for c, p in zip(labels, probs)},
    }
    if with_cam:
        top_idx = max(range(1, len(labels)), key=lambda i: probs[i])  # skip Disease_Risk
        cam = GradCAM(model)(x.clone().requires_grad_(True), class_idx=top_idx)
        out["cam_overlay"] = overlay_cam(image, cam)
        out["cam_class"] = labels[top_idx]
        out["cam_class_name"] = DISEASE_NAMES.get(labels[top_idx], labels[top_idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, labels = load_model(args.checkpoint, device)
    result = predict_image(model, labels, Image.open(args.image), device, args.top_k)

    print(f"Overall disease risk: {result['disease_risk']:.3f}")
    print("Top predictions:")
    for code, name, p in result["top"]:
        print(f"  {code:6s} {name:40s} {p:.3f}")


if __name__ == "__main__":
    main()
