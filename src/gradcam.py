import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms


class GradCAM:
    """Grad-CAM for a CNN. Default target layer is ResNet's layer4."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.layer4
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._fwd)
        self.target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, output):
        self.activations = output.detach()

    def _bwd(self, _m, _gi, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def _jet(cam: np.ndarray) -> np.ndarray:
    # simple jet-like colormap without matplotlib dependency at inference time
    import matplotlib
    return (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)


def overlay_cam(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = pil_image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    img_arr = np.array(img).astype(np.float32)
    heat = _jet(cam).astype(np.float32)
    blend = (1 - alpha) * img_arr + alpha * heat
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    return Image.fromarray(blend)


def prepare_tensor(pil_image: Image.Image, device: str) -> torch.Tensor:
    tfm = build_transforms(train=False)
    return tfm(pil_image.convert("RGB")).unsqueeze(0).to(device)
