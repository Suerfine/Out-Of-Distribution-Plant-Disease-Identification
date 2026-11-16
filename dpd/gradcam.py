"""Pure PyTorch + NumPy + PIL Grad-CAM interpretability engine.

Generates class activation maps for convolutional backbones (ResNet, ConvNeXt, EfficientNet, etc.)
without external dependencies (no opencv-python or matplotlib required).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def colormap_jet(gray: np.ndarray) -> np.ndarray:
    """Vectorised Jet colormap in pure NumPy.

    Args:
        gray: 2D float array in [0, 1] of shape (H, W).

    Returns:
        RGB uint8 array of shape (H, W, 3) in [0, 255].
    """
    x = np.clip(gray, 0.0, 1.0) * 4.0
    r = np.clip(np.minimum(x - 1.5, 4.5 - x), 0.0, 1.0)
    g = np.clip(np.minimum(x - 0.5, 3.5 - x), 0.0, 1.0)
    b = np.clip(np.minimum(x + 0.5, 2.5 - x), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_cam(orig_image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a 2D [0, 1] CAM heatmap on a PIL Image with specified blending opacity."""
    w, h = orig_image.size
    heatmap_rgb = colormap_jet(cam)
    heatmap_pil = Image.fromarray(heatmap_rgb).resize((w, h), Image.Resampling.BILINEAR)
    orig_rgb = orig_image.convert("RGB")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return Image.blend(orig_rgb, heatmap_pil, alpha=alpha)


def find_target_layer(model: nn.Module) -> nn.Module | None:
    """Locates the final spatial 2D convolutional / feature stage in a backbone.

    Supports:
    - torchvision ResNet: model.layer4[-1]
    - timm ResNet / ResNetV2: stages[-1] / layer4[-1]
    - torchvision / timm ConvNeXt: stages[-1] or features[-1]
    - torchvision / timm EfficientNet / EfficientNetV2: conv_head or features[-1] or blocks[-1]
    - Fallback: deepest nn.Conv2d module in the network.
    """
    # 1. Known common attribute paths in popular backbones
    paths = [
        "stages[-1]",
        "layer4[-1]",
        "layer4",
        "conv_head",
        "features[-1]",
        "blocks[-1]",
    ]
    for p in paths:
        try:
            curr = model
            for part in p.split("."):
                if "[" in part:
                    attr, idx = part[:-1].split("[")
                    curr = getattr(curr, attr)[int(idx)]
                else:
                    curr = getattr(curr, part)
            if isinstance(curr, nn.Module):
                return curr
        except (AttributeError, IndexError, KeyError, TypeError):
            continue

    # 2. Fallback: inspect named modules in reverse to find the last Conv2d
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    return None


class GradCamHook:
    """Context-managed forward and backward hooks for capturing activations and gradients."""

    def __init__(self, target_layer: nn.Module):
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_handle = None
        self._bwd_handle = None

    def __enter__(self):
        def fwd_hook(module, inp, out):
            val = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(val, torch.Tensor):
                self.activations = val.detach()

        def bwd_hook(module, grad_in, grad_out):
            val = grad_out[0] if isinstance(grad_out, (tuple, list)) else grad_out
            if isinstance(val, torch.Tensor):
                self.gradients = val.detach()

        self._fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()


def compute_cam(
    net: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    head_name: str | None = None,
    custom_target_fn: Any | None = None,
) -> np.ndarray:
    """Computes a 2D normalized Grad-CAM heatmap in [0, 1].

    Args:
        net: The neural network (in eval mode).
        target_layer: The target convolutional layer/block.
        input_tensor: 4D tensor (1, 3, H, W).
        target_class: Target class index to backpropagate from.
        head_name: If net outputs a dict of heads, key name to select.
        custom_target_fn: Optional callable (output_dict_or_tensor) -> scalar score.

    Returns:
        2D numpy array of shape (H, W) in [0.0, 1.0].
    """
    device = input_tensor.device
    was_training = net.training
    net.eval()

    with GradCamHook(target_layer) as hook:
        net.zero_grad(set_to_none=True)
        # Enable gradients for backward pass even if called inside torch.no_grad()
        with torch.enable_grad():
            x = input_tensor.clone().detach().requires_grad_(True)
            out = net(x)

            if custom_target_fn is not None:
                score = custom_target_fn(out)
            elif isinstance(out, dict):
                logits = out[head_name]
                score = logits[0, target_class]
            elif isinstance(out, torch.Tensor):
                score = out[0, target_class]
            else:
                raise TypeError(f"Unexpected network output type: {type(out)}")

            score.backward(retain_graph=False)

        activations = hook.activations
        gradients = hook.gradients

    if was_training:
        net.train()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

    # Spatial global average pooling of gradients: (B, C, 1, 1) or (B, C, 1) depending on tensor dims
    if gradients.ndim == 4:
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        # Upsample to input resolution
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].cpu().float().numpy()
    elif gradients.ndim == 3:  # (B, L, C)
        # Token-based feature maps: reshape if spatial dimensions are known, else fallback
        weights = torch.mean(gradients, dim=1, keepdim=True)
        cam = torch.sum(weights * activations, dim=-1)
        cam = F.relu(cam)
        cam_np = cam[0].cpu().float().numpy()
        side = int(np.sqrt(cam_np.shape[0]))
        if side * side == cam_np.shape[0]:
            cam_np = cam_np.reshape((side, side))
            cam_t = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0)
            cam_t = F.interpolate(cam_t, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
            cam_np = cam_t[0, 0].numpy()
    else:
        raise RuntimeError(f"Unsupported activation tensor rank: {activations.ndim}")

    # Min-max normalization
    c_min, c_max = float(cam_np.min()), float(cam_np.max())
    if c_max - c_min > 1e-8:
        cam_np = (cam_np - c_min) / (c_max - c_min)
    else:
        cam_np = np.zeros_like(cam_np)

    return cam_np


def explain_cnn(
    net: nn.Module,
    transform: Any,
    resize_cache: int,
    src: Any,
    pred_crop_id: int,
    pred_disease_id: int,
    is_multi_task: bool = True,
    crop_net: nn.Module | None = None,
    disease_net: nn.Module | None = None,
) -> dict[str, Any]:
    """Generates Grad-CAM explanations for Multi-Task or Independent CNN models.

    Args:
        net: Primary network (multitask_net or crop_net).
        transform: Image evaluation transform.
        resize_cache: Resize resolution for standardisation.
        src: File path, bytes, or file-like object of the original image.
        pred_crop_id: Predicted crop category index.
        pred_disease_id: Predicted disease category index.
        is_multi_task: True if multi-task (shared backbone), False if independent.
        crop_net: Dedicated crop network if independent.
        disease_net: Dedicated disease network if independent.

    Returns:
        Dictionary with explanation results, CAM arrays, overlays, and descriptions.
    """
    from dpd.resize_cache import standardise

    std_img = standardise(src, resize_cache)
    orig_pil = Image.open(src if not isinstance(src, bytes) else io.BytesIO(src)).convert("RGB")
    device = next(net.parameters()).device
    x = transform(std_img).unsqueeze(0).to(device)

    cams: dict[str, dict[str, Any]] = {}

    if is_multi_task:
        # Multi-task CNN: shared backbone
        backbone = getattr(net, "backbone", net)
        target_layer = find_target_layer(backbone)
        if target_layer is None:
            return {
                "supported": False,
                "reason": "Could not identify a 2D convolutional feature layer in this backbone.",
                "cams": {},
            }

        # 1. Disease Attention
        dis_cam = compute_cam(net, target_layer, x, target_class=pred_disease_id, head_name="disease")
        cams["disease"] = {
            "cam": dis_cam,
            "title": "Disease Attention",
            "desc": "Highlights lesions, spots, and pathology patterns used to classify the disease.",
        }

        # 2. Crop Attention
        crop_cam = compute_cam(net, target_layer, x, target_class=pred_crop_id, head_name="crop")
        cams["crop"] = {
            "cam": crop_cam,
            "title": "Crop Attention",
            "desc": "Highlights leaf shape, venation, and structural features used to identify the host plant.",
        }

        # 3. Combined Pair Attention
        def pair_target(out_dict):
            return out_dict["crop"][0, pred_crop_id] + out_dict["disease"][0, pred_disease_id]

        pair_cam = compute_cam(net, target_layer, x, custom_target_fn=pair_target)
        cams["combined"] = {
            "cam": pair_cam,
            "title": "Combined Attention",
            "desc": "Joint attention over both crop and disease target logits.",
        }

    else:
        # Independent CNN: two separate networks
        cn = crop_net or net
        dn = disease_net
        if dn is None:
            return {
                "supported": False,
                "reason": "Independent CNN requires both crop and disease networks.",
                "cams": {},
            }

        target_layer_c = find_target_layer(getattr(cn, "backbone", cn))
        target_layer_d = find_target_layer(getattr(dn, "backbone", dn))

        if target_layer_c is None or target_layer_d is None:
            return {
                "supported": False,
                "reason": "Could not find a convolutional feature layer in the independent networks.",
                "cams": {},
            }

        # Crop Net CAM
        crop_cam = compute_cam(cn, target_layer_c, x, target_class=pred_crop_id, head_name="crop" if hasattr(cn, "heads") else None)
        cams["crop"] = {
            "cam": crop_cam,
            "title": "Crop Attention (Crop Net)",
            "desc": "Highlights leaf morphology and structural features in the dedicated crop network.",
        }

        # Disease Net CAM
        dis_cam = compute_cam(dn, target_layer_d, x, target_class=pred_disease_id, head_name="disease" if hasattr(dn, "heads") else None)
        cams["disease"] = {
            "cam": dis_cam,
            "title": "Disease Attention (Disease Net)",
            "desc": "Highlights symptoms and lesions in the dedicated disease network.",
        }

    return {
        "supported": True,
        "reason": None,
        "orig_image": orig_pil,
        "cams": cams,
    }
