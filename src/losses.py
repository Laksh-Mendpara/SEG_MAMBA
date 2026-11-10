"""
Loss functions:
  L_total = CE(Z_seg, Y) + Dice(Z_seg, Y) + 0.2 * BCE_edge(Z_edge, ∂Y)

GT edge map generated on-the-fly via morphological max-pooling (no extra annotations).
All losses are computed in float32 for mixed-precision numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# GT Edge Map
# ════════════════════════════════════════════════════════════════════════════

def get_gt_edges(labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    Generate binary GT edge map on-the-fly via morphological dilation.

    A pixel is an edge if its class label differs from any neighbour in a
    3×3 window. Border/ignore pixels do NOT contribute.

    Args:
        labels: [B, H, W] int64 — ground-truth class labels
        ignore_index: pixels to exclude

    Returns:
        edges: [B, 1, H, W] float32 ∈ {0, 1}
    """
    # Temporarily replace ignore pixels with -1 so they don't blur edges
    lf = labels.float().clone()
    lf[labels == ignore_index] = -1.0
    lf = lf.unsqueeze(1)                      # [B, 1, H, W]

    # Dilate via max-pool: if any neighbour has a different valid label → edge
    dilated = F.max_pool2d(lf, kernel_size=3, stride=1, padding=1)
    edge = (dilated != lf).float()

    # Zero-out ignore pixels in the edge map
    edge[labels.unsqueeze(1) == ignore_index] = 0.0
    return edge                                 # [B, 1, H, W]


# ════════════════════════════════════════════════════════════════════════════
# CE + Dice + Edge BCE composite loss
# ════════════════════════════════════════════════════════════════════════════

class SegmentationLoss(nn.Module):
    """
    Three-term composite segmentation loss.

    L_total = w_ce  * L_CE(Z_seg, Y)
            + w_dice* L_Dice(Z_seg, Y)
            + w_edge* L_BCE(Z_edge, ∂Y)
    """

    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: int = 255,
        w_ce:   float = 1.0,
        w_dice: float = 1.0,
        w_edge: float = 0.2,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.w_ce   = w_ce
        self.w_dice = w_dice
        self.w_edge = w_edge

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    # ── Dice ─────────────────────────────────────────────────────────────

    def _dice_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Multi-class soft Dice loss.

        L_Dice = 1 - (1/K_present) Σ_{k ∈ present} 2*inter / (denom)
        Vectorized across all classes and computed in float32 for numerical stability.
        Only evaluates classes present in the batch ground-truth target to prevent
        empty-class dilution/instability in large label sets (Cityscapes, ADE20K).
        """
        B, K, H, W = logits.shape
        probs = torch.softmax(logits.float(), dim=1)      # compute in fp32

        # One-hot target, ignore pixels zeroed
        valid_mask = (labels != self.ignore_index)        # [B, H, W]
        labels_clamped = labels.clone()
        labels_clamped[~valid_mask] = 0
        one_hot = F.one_hot(labels_clamped, K).permute(0, 3, 1, 2).float()  # [B, K, H, W]
        one_hot = one_hot * valid_mask.unsqueeze(1).float()

        eps = 1e-6
        p_vec = probs.view(B, K, -1)                      # [B, K, HW]
        t_vec = one_hot.view(B, K, -1)                    # [B, K, HW]

        inter = (p_vec * t_vec).sum(dim=2)               # [B, K]
        denom = (p_vec * p_vec).sum(dim=2) + (t_vec * t_vec).sum(dim=2)  # [B, K]
        dice = (2.0 * inter + eps) / (denom + eps)       # [B, K]

        # Filter to classes actually present in the batch targets
        present = (t_vec.sum(dim=2) > 0)                 # [B, K]
        if present.sum() > 0:
            return 1.0 - dice[present].mean()
        return logits.sum() * 0.0

    # ── Weighted edge BCE ─────────────────────────────────────────────────

    def _edge_bce_loss(self, edge_pred: torch.Tensor, gt_edge: torch.Tensor) -> torch.Tensor:
        """
        Weighted binary cross-entropy for edge supervision.
        pos_weight = N_neg / N_pos  (~9–10x to handle edge sparsity).
        """
        n_pos = gt_edge.sum().clamp(min=1.0)
        n_neg = (1.0 - gt_edge).sum().clamp(min=1.0)
        pos_weight = (n_neg / n_pos).clamp(max=100.0).detach()
        return F.binary_cross_entropy_with_logits(
            edge_pred.float(), gt_edge.float(), pos_weight=pos_weight, reduction="mean"
        )

    # ── Combined ──────────────────────────────────────────────────────────

    def forward(
        self,
        logits: torch.Tensor,       # [B, K, H, W]
        edge_pred: torch.Tensor,    # [B, 1, H, W]
        labels: torch.Tensor,       # [B, H, W] int64
    ) -> dict:
        valid_pixels = (labels != self.ignore_index).sum()
        if valid_pixels == 0:
            zero_val = logits.sum() * 0.0
            return {
                "total": zero_val,
                "ce":    zero_val.detach(),
                "dice":  zero_val.detach(),
                "edge":  zero_val.detach(),
            }

        gt_edge = get_gt_edges(labels, self.ignore_index)

        # Compute losses in float32 for AMP stability
        l_ce = self.ce_loss(logits.float(), labels)
        if torch.isnan(l_ce) or torch.isinf(l_ce):
            l_ce = logits.sum() * 0.0

        l_dice = self._dice_loss(logits, labels)
        if torch.isnan(l_dice) or torch.isinf(l_dice):
            l_dice = logits.sum() * 0.0

        l_edge = self._edge_bce_loss(edge_pred, gt_edge)
        if torch.isnan(l_edge) or torch.isinf(l_edge):
            l_edge = logits.sum() * 0.0

        total = self.w_ce * l_ce + self.w_dice * l_dice + self.w_edge * l_edge
        return {
            "total": total,
            "ce":    l_ce.detach(),
            "dice":  l_dice.detach(),
            "edge":  l_edge.detach(),
        }


# ════════════════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════════════════

def build_loss(cfg: dict) -> SegmentationLoss:
    weights = cfg["training"].get("loss_weights", {})
    return SegmentationLoss(
        num_classes  = cfg["model"]["num_classes"],
        ignore_index = cfg["data"].get("ignore_index", 255),
        w_ce         = weights.get("ce",   1.0),
        w_dice       = weights.get("dice", 1.0),
        w_edge       = weights.get("edge", 0.2),
    )
