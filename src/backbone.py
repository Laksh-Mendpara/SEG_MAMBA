"""
Backbone wrappers — returns {F1, F2, F3, F4} multi-scale feature maps
using timm's features_only API.

Supported:
  - SwinTinyBackbone  (swin_tiny_patch4_window7_224)   channels: (96, 192, 384, 768)
  - SwinBaseBackbone  (swin_base_patch4_window12_384)  channels: (128, 256, 512, 1024)

Both use dynamic_img_size=True to support arbitrary input resolutions.
"""

import torch
import torch.nn as nn
import timm


class SwinTinyBackbone(nn.Module):
    """
    Swin-Tiny backbone.
    Output channels per stage: (96, 192, 384, 768) at strides (4, 8, 16, 32).
    """

    OUT_CHANNELS = (96, 192, 384, 768)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            dynamic_img_size=True,
            out_indices=(0, 1, 2, 3),
        )
        self.out_channels = self.OUT_CHANNELS

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        out = []
        for f in feats:
            if f.dim() == 4 and f.shape[-1] != f.shape[1]:
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out


class SwinBaseBackbone(nn.Module):
    """
    Swin-Base backbone (ImageNet-22K pretrained).
    Output channels per stage: (128, 256, 512, 1024) at strides (4, 8, 16, 32).

    Recommended for Cityscapes (19 cls) and ADE20K (150 cls) where richer
    encoder features substantially improve multi-class mIoU.
    Uses window12 variant to better handle 768x768 / 512x512 inputs.
    """

    OUT_CHANNELS = (128, 256, 512, 1024)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window12_384",
            pretrained=pretrained,
            features_only=True,
            dynamic_img_size=True,
            out_indices=(0, 1, 2, 3),
        )
        self.out_channels = self.OUT_CHANNELS

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        out = []
        for f in feats:
            if f.dim() == 4 and f.shape[-1] != f.shape[1]:
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out


# ─────────────────────────── factory ────────────────────────────────────────

_BACKBONE_REGISTRY = {
    "swin_tiny_patch4_window7_224":  SwinTinyBackbone,
    "swin_base_patch4_window12_384": SwinBaseBackbone,
}


def build_backbone(cfg: dict) -> nn.Module:
    name       = cfg["model"].get("backbone", "swin_tiny_patch4_window7_224")
    pretrained = cfg["model"].get("backbone_pretrained", True)
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {list(_BACKBONE_REGISTRY.keys())}"
        )
    return _BACKBONE_REGISTRY[name](pretrained=pretrained)
