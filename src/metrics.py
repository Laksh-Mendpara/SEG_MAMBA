"""
Evaluation metrics:
  - mIoU  (Mean Intersection-over-Union)
  - Pixel Accuracy
  - Boundary IoU (BIoU) — IoU computed only within d pixels of GT boundary
"""

import torch
import torch.nn.functional as F
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# Confusion-matrix-based mIoU + pixel accuracy
# ════════════════════════════════════════════════════════════════════════════

class SegmentationMetrics:
    """
    Accumulates predictions over the full val set then computes metrics.

    Usage:
        metrics = SegmentationMetrics(num_classes=2, ignore_index=255)
        for batch:
            pred = model(x).argmax(1)   # [B, H, W]
            metrics.update(pred, label) # [B, H, W]
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = 255):
        self.K = num_classes
        self.ignore_index = ignore_index
        self.conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.conf_matrix[:] = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred:   [B, H, W] int64 — predicted class indices
            target: [B, H, W] int64 — GT labels (ignore_index pixels skipped)
        """
        pred   = pred.cpu().numpy().astype(np.int64).ravel()
        target = target.cpu().numpy().astype(np.int64).ravel()

        valid  = target != self.ignore_index
        pred   = pred[valid]
        target = target[valid]

        # Clamp predictions to valid range (safety)
        pred = np.clip(pred, 0, self.K - 1)

        # Accumulate confusion matrix
        idx = target * self.K + pred
        cnt = np.bincount(idx, minlength=self.K * self.K)
        self.conf_matrix += cnt.reshape(self.K, self.K)

    def compute(self) -> dict:
        cm = self.conf_matrix.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp

        iou_per_class = tp / (tp + fp + fn + 1e-10)
        miou  = np.mean(iou_per_class)
        pix_acc = tp.sum() / (cm.sum() + 1e-10)

        return {
            "miou":          float(miou),
            "pixel_acc":     float(pix_acc),
            "iou_per_class": iou_per_class.tolist(),
        }


# ════════════════════════════════════════════════════════════════════════════
# Boundary IoU (BIoU)
# ════════════════════════════════════════════════════════════════════════════

def batch_boundary_iou(pred: torch.Tensor, target: torch.Tensor,
                       num_classes: int = 2, ignore_index: int = 255,
                       dilation_ratio: float = 0.02) -> torch.Tensor:
    """
    Batched GPU computation of Boundary IoU.
    Only computes boundary ribbons for classes actively present in the batch,
    yielding a ~5–15x speedup for 19–150 class tasks without altering metric values.

    pred, target: [B, H, W]
    Returns: [B] tensor of mean Boundary IoU per image across valid classes.
    """
    B, H, W = pred.shape
    d = max(1, int(round(dilation_ratio * (H ** 2 + W ** 2) ** 0.5)))
    ks = 2 * d + 1

    valid_mask = (target != ignore_index)
    valid_p = pred[valid_mask]
    valid_t = target[valid_mask]
    if valid_p.numel() == 0 or valid_t.numel() == 0:
        return torch.zeros(B, device=pred.device)

    active_classes = torch.unique(torch.cat([valid_p, valid_t]))
    if len(active_classes) == 0:
        return torch.zeros(B, device=pred.device)

    class_ious = []
    for c in active_classes:
        c_val = c.item()
        p_c = (pred == c_val).float().unsqueeze(1)    # [B, 1, H, W]
        t_c = (target == c_val).float().unsqueeze(1)  # [B, 1, H, W]

        # Dilate and erode to extract boundary ribbon
        p_dil = F.max_pool2d(p_c, ks, stride=1, padding=d)
        p_ero = -F.max_pool2d(-p_c, ks, stride=1, padding=d)
        p_bound = (p_dil != p_ero) & valid_mask.unsqueeze(1)

        t_dil = F.max_pool2d(t_c, ks, stride=1, padding=d)
        t_ero = -F.max_pool2d(-t_c, ks, stride=1, padding=d)
        t_bound = (t_dil != t_ero) & valid_mask.unsqueeze(1)

        inter = (p_bound & t_bound).sum(dim=(1, 2, 3)).float()
        union = (p_bound | t_bound).sum(dim=(1, 2, 3)).float()
        has_gt = t_c.sum(dim=(1, 2, 3)) > 0

        iou = torch.where(
            union > 0,
            inter / union,
            torch.where(has_gt, torch.zeros_like(union), torch.full_like(union, float('nan')))
        )
        class_ious.append(iou)

    stacked = torch.stack(class_ious, dim=1)  # [B, K_active]
    # Average across non-NaN classes per image
    nan_mask = torch.isnan(stacked)
    stacked[nan_mask] = 0.0
    valid_counts = (~nan_mask).sum(dim=1).clamp(min=1).float()
    return stacked.sum(dim=1) / valid_counts  # [B]


class BoundaryIoUMetric:
    """Accumulates per-image BIoU on GPU/CPU then averages."""

    def __init__(self, num_classes: int = 2, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.scores = []

    def reset(self):
        self.scores.clear()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """pred, target: [B, H, W] on device (CUDA or CPU)"""
        b_scores = batch_boundary_iou(
            pred, target, self.num_classes, self.ignore_index
        )
        self.scores.extend(b_scores.detach().cpu().tolist())

    def compute(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0
