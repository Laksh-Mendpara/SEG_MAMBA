"""
Utilities:
  - Config loading
  - Run directory management
  - Loss curve plotting
  - Multi-dataset Segmentation visualization
  - Checkpoint saving/loading
  - Logger setup
  - Run report generation
"""

import os
import sys
import json
import logging
import datetime
import colorsys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import yaml


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ════════════════════════════════════════════════════════════════════════════
# Run directory
# ════════════════════════════════════════════════════════════════════════════

def make_run_dir(base: str, exp_name: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base) / f"{ts}_{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    return run_dir


# ════════════════════════════════════════════════════════════════════════════
# Logger
# ════════════════════════════════════════════════════════════════════════════

def setup_logger(run_dir: Path, name: str = "sdbc") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ════════════════════════════════════════════════════════════════════════════
# Metric tracker
# ════════════════════════════════════════════════════════════════════════════

class MetricTracker:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.history: dict[str, list] = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.history.setdefault(k, []).append(float(v))

    def save(self):
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def load(self):
        p = self.run_dir / "history.json"
        if p.exists():
            with open(p) as f:
                self.history = json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# Loss curve plot
# ════════════════════════════════════════════════════════════════════════════

def plot_losses(history: dict, run_dir: Path, epoch: int):
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [
        ("train_loss_total", "val_loss_total", "Total Loss"),
        ("train_miou",       "val_miou",       "mIoU"),
        ("val_biou",         None,             "Boundary IoU"),
    ]

    for ax, (train_key, val_key, title) in zip(axes, pairs):
        if train_key in history:
            ax.plot(history[train_key], label="train", linewidth=1.5)
        if val_key and val_key in history:
            ax.plot(history[val_key], label="val", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / f"curves_epoch{epoch:04d}.png", dpi=100, bbox_inches="tight")
    plt.savefig(plots_dir / "curves_latest.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_loss_components(history: dict, run_dir: Path, epoch: int):
    plots_dir = run_dir / "plots"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = [("train_loss_ce", "CE Loss"), ("train_loss_dice", "Dice Loss"), ("train_loss_edge", "Edge Loss")]
    for ax, (k, title) in zip(axes, keys):
        if k in history:
            ax.plot(history[k], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"loss_components_epoch{epoch:04d}.png", dpi=100, bbox_inches="tight")
    plt.savefig(plots_dir / "loss_components_latest.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Palettes & Color Mapping
# ════════════════════════════════════════════════════════════════════════════

_PET_PALETTE = {
    0: (64, 64, 64),     # Background
    1: (0, 200, 100),    # Pet foreground
}

CITYSCAPES_19_PALETTE = {
    0:  (128, 64, 128),  # road
    1:  (244, 35, 232),  # sidewalk
    2:  (70, 70, 70),    # building
    3:  (102, 102, 156), # wall
    4:  (190, 153, 153), # fence
    5:  (153, 153, 153), # pole
    6:  (250, 170, 30),  # traffic light
    7:  (220, 220, 0),   # traffic sign
    8:  (107, 142, 35),  # vegetation
    9:  (152, 251, 152), # terrain
    10: (70, 130, 180),  # sky
    11: (220, 20, 60),   # person
    12: (255, 0, 0),     # rider
    13: (0, 0, 142),     # car
    14: (0, 0, 70),      # truck
    15: (0, 60, 100),    # bus
    16: (0, 80, 100),    # train
    17: (0, 0, 230),     # motorcycle
    18: (119, 11, 32),   # bicycle
}

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
]

# High-contrast deterministic palette for ADE20K 150 classes
def _build_ade20k_palette():
    palette = {}
    for i in range(150):
        # Golden ratio hue distribution ensures maximal perceptual distance
        h = (i * 0.618033988749895) % 1.0
        s = 0.65 + 0.35 * ((i % 5) / 4.0)
        v = 0.75 + 0.25 * (((i // 5) % 4) / 3.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        palette[i] = (int(r * 255), int(g * 255), int(b * 255))
    return palette

_ADE20K_PALETTE = _build_ade20k_palette()

ADE20K_TOP_CLASSES = [
    (0, "wall"), (1, "building"), (2, "sky"), (3, "floor"), (4, "tree"),
    (5, "ceiling"), (6, "road"), (7, "bed"), (8, "window"), (9, "grass"),
    (10, "cabinet"), (11, "sidewalk"), (12, "person"), (13, "earth"), (14, "door")
]


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization → uint8 HWC numpy."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _mask_to_rgb(mask: np.ndarray, num_classes: int = 2) -> np.ndarray:
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    if num_classes == 2:
        palette = _PET_PALETTE
        void_color = (220, 190, 40) # amber/yellow border
    elif num_classes == 19:
        palette = CITYSCAPES_19_PALETTE
        void_color = (0, 0, 0)       # black void / ignore
    else:
        palette = _ADE20K_PALETTE
        void_color = (0, 0, 0)

    for cls, color in palette.items():
        rgb[mask == cls] = color

    rgb[mask == 255] = void_color
    return rgb


def _create_overlay(img_rgb: np.ndarray, mask: np.ndarray, num_classes: int = 2, alpha: float = 0.45) -> np.ndarray:
    overlay = img_rgb.copy()
    mask_rgb = _mask_to_rgb(mask, num_classes)
    valid = (mask != 255) if num_classes != 2 else (mask == 1)
    overlay[valid] = (overlay[valid] * (1.0 - alpha) + mask_rgb[valid] * alpha).astype(np.uint8)
    return overlay


def _create_gt_overlay(img_rgb: np.ndarray, gt: np.ndarray, num_classes: int = 2, alpha: float = 0.45) -> np.ndarray:
    overlay = img_rgb.copy()
    gt_rgb = _mask_to_rgb(gt, num_classes)
    valid = (gt != 255)
    overlay[valid] = (overlay[valid] * (1.0 - alpha) + gt_rgb[valid] * alpha).astype(np.uint8)
    if num_classes == 2:
        border_color = np.array([255, 190, 30], dtype=np.uint8)
        border_idx = (gt == 255) | (gt == 3)
        overlay[border_idx] = (overlay[border_idx] * (1.0 - alpha) + border_color * alpha).astype(np.uint8)
    return overlay


# ════════════════════════════════════════════════════════════════════════════
# Save Segmentation Visualizations
# ════════════════════════════════════════════════════════════════════════════

def save_segmentation_vis(
    images: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    run_dir: Path,
    epoch: int,
    edge_preds: torch.Tensor | None = None,
    stems: list[str] | None = None,
    n: int = 8,
    num_classes: int = 2,
):
    vis_base = run_dir / "visualizations"
    epoch_dir = vis_base / f"epoch-{epoch:04d}"
    examples_dir = epoch_dir / "examples"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    latest_dir = vis_base / "latest"
    latest_examples = latest_dir / "examples"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_examples.mkdir(parents=True, exist_ok=True)

    n = min(n, images.shape[0])

    for i in range(n):
        ex_dir = examples_dir / f"example-{i+1}"
        ex_dir.mkdir(exist_ok=True)
        latest_ex_dir = latest_examples / f"example-{i+1}"
        latest_ex_dir.mkdir(exist_ok=True)

        img_rgb = _denorm(images[i])
        pred_np = preds[i].cpu().numpy().astype(np.int64)
        gt_np   = labels[i].cpu().numpy().astype(np.int64)

        pred_rgb = _mask_to_rgb(pred_np, num_classes)
        gt_rgb   = _mask_to_rgb(gt_np, num_classes)
        pred_overlay = _create_overlay(img_rgb, pred_np, num_classes)
        gt_overlay   = _create_gt_overlay(img_rgb, gt_np, num_classes)

        Image.fromarray(img_rgb).save(ex_dir / "actual_image.png")
        Image.fromarray(pred_rgb).save(ex_dir / "prediction.png")
        Image.fromarray(gt_rgb).save(ex_dir / "reference_ground_truth.png")
        Image.fromarray(pred_overlay).save(ex_dir / "overlay.png")
        Image.fromarray(gt_overlay).save(ex_dir / "overlay_ground_truth.png")

        Image.fromarray(img_rgb).save(latest_ex_dir / "actual_image.png")
        Image.fromarray(pred_rgb).save(latest_ex_dir / "prediction.png")
        Image.fromarray(gt_rgb).save(latest_ex_dir / "reference_ground_truth.png")
        Image.fromarray(pred_overlay).save(latest_ex_dir / "overlay.png")
        Image.fromarray(gt_overlay).save(latest_ex_dir / "overlay_ground_truth.png")

        if edge_preds is not None and i < edge_preds.shape[0]:
            edge_prob = torch.sigmoid(edge_preds[i, 0]).cpu().numpy()
            edge_uint8 = (np.clip(edge_prob, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(edge_uint8).save(ex_dir / "edge_prediction.png")
            Image.fromarray(edge_uint8).save(latest_ex_dir / "edge_prediction.png")

        valid = (gt_np != 255)
        pix_acc = float((pred_np[valid] == gt_np[valid]).mean()) if valid.sum() > 0 else 0.0

        example_meta = {
            "example_index": i + 1,
            "stem": stems[i] if stems and i < len(stems) else f"sample_{i+1}",
            "pixel_accuracy_percent": round(pix_acc * 100, 2),
            "valid_pixels": int(valid.sum()),
            "ignored_pixels": int((gt_np == 255).sum()),
        }
        with open(ex_dir / "metrics.json", "w") as f:
            json.dump(example_meta, f, indent=2)
        with open(latest_ex_dir / "metrics.json", "w") as f:
            json.dump(example_meta, f, indent=2)

    # 4-column overview grid
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        img_rgb = _denorm(images[i])
        pred_np = preds[i].cpu().numpy()
        gt_np   = labels[i].cpu().numpy()

        stem_str = stems[i] if stems and i < len(stems) else f"Sample {i+1}"
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Image ({stem_str})", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(_create_overlay(img_rgb, pred_np, num_classes))
        axes[i, 1].set_title("Prediction Overlay", fontsize=11)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(_mask_to_rgb(pred_np, num_classes))
        axes[i, 2].set_title("Predicted Mask", fontsize=11)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(_mask_to_rgb(gt_np, num_classes))
        axes[i, 3].set_title("Reference (Ground Truth)", fontsize=11)
        axes[i, 3].axis("off")

    # Legend
    if num_classes == 2:
        patches = [
            mpatches.Patch(color=[c/255 for c in col], label=lbl)
            for lbl, col in [
                ("Background", (50, 50, 55)),
                ("Pet (Foreground)", (0, 215, 115)),
                ("Boundary Ribbon (Ignore)", (220, 190, 40))
            ]
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11, frameon=True)
    elif num_classes == 19:
        patches = [
            mpatches.Patch(color=[c/255 for c in CITYSCAPES_19_PALETTE[idx]], label=name)
            for idx, name in enumerate(CITYSCAPES_CLASS_NAMES)
        ]
        fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=8, frameon=True)
    elif num_classes == 150:
        patches = [
            mpatches.Patch(color=[c/255 for c in _ADE20K_PALETTE[idx]], label=name)
            for idx, name in ADE20K_TOP_CLASSES
        ]
        fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8, frameon=True,
                   title="ADE20K (Top 15 Classes)")

    plt.suptitle(f"Evaluation Visualizations — Epoch {epoch}", fontsize=15, y=1.005)
    plt.tight_layout()

    grid_path = epoch_dir / "comparison_grid.png"
    plt.savefig(grid_path, dpi=100, bbox_inches="tight")
    plt.savefig(latest_dir / "comparison_grid.png", dpi=100, bbox_inches="tight")
    plt.savefig(vis_base / "vis_latest.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    return grid_path


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint & Report
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(state: dict, run_dir: Path, filename: str):
    path = run_dir / "checkpoints" / filename
    torch.save(state, path)
    return path


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)


def save_run_report(
    run_dir: Path,
    cfg: dict,
    model: torch.nn.Module,
    tracker: MetricTracker,
    start_time: float,
    end_time: float,
    status: str = "COMPLETED",
    best_miou: float = 0.0,
    best_epoch: int = 0,
) -> Path:
    total_sec = max(0.0, end_time - start_time)
    hours, rem = divmod(total_sec, 3600)
    mins, secs = divmod(rem, 60)
    dur_str = f"{int(hours):02d}h {int(mins):02d}m {int(secs):02d}s"

    history = tracker.history
    total_epochs = len(history.get("train_loss_total", []))
    avg_epoch_time = total_sec / max(total_epochs, 1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    val_miou_list = history.get("val_miou", [])
    val_biou_list = history.get("val_biou", [])
    val_loss_list = history.get("val_loss_total", [])
    train_loss_list = history.get("train_loss_total", [])
    val_acc_list = history.get("val_pixel_acc", [])

    if val_miou_list:
        max_miou = max(val_miou_list)
        best_ep = val_miou_list.index(max_miou) + 1
        biou_at_best = val_biou_list[best_ep - 1] if len(val_biou_list) >= best_ep else 0.0
        acc_at_best = val_acc_list[best_ep - 1] if len(val_acc_list) >= best_ep else 0.0
    else:
        max_miou = best_miou
        best_ep = best_epoch
        biou_at_best = 0.0
        acc_at_best = 0.0

    report = {
        "experiment_name": run_dir.name,
        "status": status,
        "timing": {
            "start_time": datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": round(total_sec, 2),
            "total_duration_formatted": dur_str,
            "total_epochs_completed": total_epochs,
            "avg_time_per_epoch_seconds": round(avg_epoch_time, 2),
        },
        "model_summary": {
            "architecture": "SDBC-Mamba",
            "backbone": cfg["model"].get("backbone", "swin_tiny_patch4_window7_224"),
            "embed_dim": cfg["model"].get("embed_dim", 128),
            "d_state": cfg["model"].get("d_state", 16),
            "expand": cfg["model"].get("expand", 2),
            "num_ssm_blocks": cfg["model"].get("num_ssm_blocks", 2),
            "total_parameters": total_params,
            "total_parameters_million": round(total_params / 1e6, 2),
            "trainable_parameters": trainable_params,
            "trainable_parameters_million": round(trainable_params / 1e6, 2),
        },
        "dataset_summary": {
            "dataset_name": cfg["data"].get("dataset", "oxford_pet"),
            "num_classes": cfg["model"].get("num_classes", 2),
            "image_size": cfg["data"].get("image_size", 512),
            "ignore_index": cfg["data"].get("ignore_index", 255),
        },
        "best_performance": {
            "best_epoch": best_ep,
            "best_val_miou_percent": round(max_miou * 100, 2),
            "best_val_biou_percent": round(biou_at_best * 100, 2),
            "best_val_pixel_acc_percent": round(acc_at_best * 100, 2),
        },
    }

    report_path = run_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report_path
