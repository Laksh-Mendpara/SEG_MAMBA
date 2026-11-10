"""
Main training script for SDBC-Mamba.
Supports Oxford-IIIT Pet, Cityscapes, and ADE20K via config.

Usage:
    python train.py --config configs/sdbc_mamba_cityscapes.yaml --exp cs_run1
    python train.py --config configs/sdbc_mamba_ade20k.yaml     --exp ade_run1
    python train.py --config configs/sdbc_mamba_pet.yaml        --exp pet_run1  [--resume ckpt.pth]
"""

import argparse
import os
import sys
import time
import shutil
import yaml
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))
from src.utils   import (
    load_config, make_run_dir, setup_logger, MetricTracker,
    plot_losses, plot_loss_components,
    save_segmentation_vis, save_checkpoint, load_checkpoint,
    save_run_report,
)
from src.dataset import build_dataloaders
from src.decoder import build_model
from src.losses  import build_loss
from src.metrics import SegmentationMetrics, BoundaryIoUMetric


# ════════════════════════════════════════════════════════════════════════════
# W&B helper (gracefully disabled if wandb unavailable / not logged in)
# ════════════════════════════════════════════════════════════════════════════

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _wandb_enabled(cfg) -> bool:
    return _WANDB_AVAILABLE and cfg.get("logging", {}).get("wandb", False)


def wandb_init(cfg, run_dir, exp_name):
    if not _wandb_enabled(cfg):
        return
    project = cfg["logging"].get(
        "wandb_project",
        f"cv_{cfg['data'].get('dataset', 'sdbc_mamba')}"
    )
    wandb.init(
        project=project,
        name=exp_name,
        config=cfg,
        dir=str(run_dir),
        resume="allow",
    )


def wandb_log(cfg, payload: dict, step: int):
    if not _wandb_enabled(cfg):
        return
    wandb.log(payload, step=step)


def wandb_log_image(cfg, caption: str, path: Path, step: int):
    if not _wandb_enabled(cfg) or not path.exists():
        return
    wandb.log({caption: wandb.Image(str(path))}, step=step)


def wandb_finish(cfg):
    if not _wandb_enabled(cfg):
        return
    wandb.finish()


# ════════════════════════════════════════════════════════════════════════════
# Warmup-Cosine LR scheduler
# ════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """Linear warmup for `warmup_epochs`, then cosine decay to eta_min."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-7):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        self._base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self._cosine       = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=eta_min,
        )
        self._epoch = 0

    def step(self):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            frac = self._epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * frac
        else:
            self._cosine.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {"epoch": self._epoch, "cosine": self._cosine.state_dict()}

    def load_state_dict(self, d):
        self._epoch = d["epoch"]
        self._cosine.load_state_dict(d["cosine"])


# ════════════════════════════════════════════════════════════════════════════
# Training epoch
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device, epoch, logger, log_interval, cfg):
    model.train()
    total_loss = ce_acc = dice_acc = edge_acc = 0.0
    n_batches = len(loader)
    amp_enabled = cfg["training"].get("amp", True) and device.type == "cuda"
    use_bf16 = amp_enabled and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for step, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            logits, edge_pred = model(images)
            losses = criterion(logits, edge_pred, labels)

        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            logger.warning(
                f"Epoch {epoch} [{step+1}/{n_batches}] NaN/Inf loss encountered "
                f"(ce={losses['ce'].item()}, dice={losses['dice'].item()}, edge={losses['edge'].item()}). "
                "Skipping step to protect model weights."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_bf16:
            losses["total"].backward()
            clip = cfg["training"].get("clip_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            clip = cfg["training"].get("clip_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += losses["total"].item()
        ce_acc     += losses["ce"].item()
        dice_acc   += losses["dice"].item()
        edge_acc   += losses["edge"].item()

        if (step + 1) % log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} [{step+1}/{n_batches}]  "
                f"loss={total_loss/(step+1):.4f}  "
                f"ce={losses['ce'].item():.4f}  "
                f"dice={losses['dice'].item():.4f}  "
                f"edge={losses['edge'].item():.4f}  "
                f"lr={lr:.2e}"
            )

    n = n_batches
    return {
        "loss_total": total_loss / n,
        "loss_ce":    ce_acc    / n,
        "loss_dice":  dice_acc  / n,
        "loss_edge":  edge_acc  / n,
    }


# ════════════════════════════════════════════════════════════════════════════
# Validation epoch
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, logger, cfg,
             run_dir, num_vis_samples):
    model.eval()
    num_classes  = cfg["model"]["num_classes"]
    ignore_index = cfg["data"].get("ignore_index", 255)

    seg_metrics  = SegmentationMetrics(num_classes, ignore_index)
    biou_metrics = BoundaryIoUMetric(num_classes, ignore_index)

    total_loss   = 0.0
    vis_images, vis_preds, vis_labels, vis_edges, vis_stems = [], [], [], [], []
    n_vis = 0

    amp_enabled = cfg["training"].get("amp", True) and device.type == "cuda"
    use_bf16 = amp_enabled and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for images, labels, stems in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            logits, edge_pred = model(images)
            losses = criterion(logits, edge_pred, labels)

        total_loss += losses["total"].item()
        preds = logits.argmax(dim=1)
        seg_metrics.update(preds, labels)
        biou_metrics.update(preds, labels)

        if n_vis < num_vis_samples:
            rem = num_vis_samples - n_vis
            vis_images.append(images[:rem].cpu())
            vis_preds.append(preds[:rem].cpu())
            vis_labels.append(labels[:rem].cpu())
            if edge_pred is not None:
                vis_edges.append(edge_pred[:rem].cpu())
            vis_stems.extend(stems[:rem])
            n_vis += min(rem, images.shape[0])

    seg_res = seg_metrics.compute()
    biou    = biou_metrics.compute()
    avg_loss = total_loss / len(loader)

    logger.info(
        f"Val Epoch {epoch} — loss={avg_loss:.4f}  "
        f"mIoU={seg_res['miou']*100:.2f}%  "
        f"BIoU={biou*100:.2f}%  "
        f"PixAcc={seg_res['pixel_acc']*100:.2f}%"
    )

    return {
        "loss_total": avg_loss,
        "miou":       seg_res["miou"],
        "biou":       biou,
        "pixel_acc":  seg_res["pixel_acc"],
        "iou_per_class": seg_res["iou_per_class"],
        "vis": (vis_images, vis_preds, vis_labels, vis_edges, vis_stems),
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train SDBC-Mamba")
    parser.add_argument("--config",  default="configs/sdbc_mamba_pet.yaml")
    parser.add_argument("--exp",     default="sdbc_mamba")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--resume",  default=None)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device)

    # ── run directory ────────────────────────────────────────────────────
    runs_base = cfg["logging"]["runs_dir"]
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("checkpoints", "plots", "visualizations"):
            (run_dir / sub).mkdir(exist_ok=True)
    else:
        run_dir = make_run_dir(runs_base, args.exp)

    logger  = setup_logger(run_dir)
    tracker = MetricTracker(run_dir)

    logger.info(f"Run dir  : {run_dir}")
    logger.info(f"Device   : {device}")
    logger.info(f"Config   : {args.config}")
    logger.info(f"Dataset  : {cfg['data'].get('dataset', 'oxford_pet')}")

    shutil.copy(args.config, run_dir / "config.yaml")

    # ── W&B init ────────────────────────────────────────────────────────
    wandb_init(cfg, run_dir, args.exp)

    # ── data ─────────────────────────────────────────────────────────────
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # ── model ─────────────────────────────────────────────────────────────
    logger.info("Building model...")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params/1e6:.2f}M")

    # ── loss / optimizer / scheduler ──────────────────────────────────────
    criterion = build_loss(cfg).to(device)
    tcfg      = cfg["training"]

    optimizer = AdamW([
        {"params": list(model.backbone.parameters()), "lr": tcfg["backbone_lr"]},
        {"params": list(model.decoder.parameters()),  "lr": tcfg["decoder_lr"]},
    ], weight_decay=tcfg["weight_decay"])

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=tcfg.get("warmup_epochs", 5),
        total_epochs=tcfg["epochs"],
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=tcfg.get("amp", True) and device.type == "cuda"
    )

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou   = 0.0
    best_epoch  = 0
    if args.resume:
        start_epoch, best_miou = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        best_epoch = start_epoch
        tracker.load()
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ── intervals ─────────────────────────────────────────────────────────
    log_interval    = cfg["logging"].get("log_interval", 20)
    vis_interval    = cfg["logging"].get("vis_interval", 1)
    save_interval   = cfg["logging"].get("save_interval", 5)
    num_vis_samples = cfg["logging"].get("num_vis_samples", 8)

    logger.info("Starting training...")
    start_time = time.time()
    status = "COMPLETED"

    try:
        for epoch in range(start_epoch + 1, tcfg["epochs"] + 1):
            t0 = time.time()

            # ── train ─────────────────────────────────────────────────
            train_stats = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler,
                device, epoch, logger, log_interval, cfg
            )

            # ── validate ──────────────────────────────────────────────
            val_stats = validate(
                model, val_loader, criterion, device, epoch, logger, cfg,
                run_dir, num_vis_samples
            )

            scheduler.step()
            elapsed = time.time() - t0

            # ── tracker ───────────────────────────────────────────────
            tracker.update(
                train_loss_total = train_stats["loss_total"],
                train_loss_ce    = train_stats["loss_ce"],
                train_loss_dice  = train_stats["loss_dice"],
                train_loss_edge  = train_stats["loss_edge"],
                val_loss_total   = val_stats["loss_total"],
                val_miou         = val_stats["miou"],
                val_biou         = val_stats["biou"],
                val_pixel_acc    = val_stats["pixel_acc"],
            )
            tracker.save()

            logger.info(
                f"Epoch {epoch}/{tcfg['epochs']} — {elapsed:.1f}s  "
                f"train={train_stats['loss_total']:.4f}  "
                f"val_mIoU={val_stats['miou']*100:.2f}%  "
                f"val_BIoU={val_stats['biou']*100:.2f}%"
            )

            # ── W&B logging ───────────────────────────────────────────
            lr_now = optimizer.param_groups[0]["lr"]
            wandb_payload = {
                "train/loss":      train_stats["loss_total"],
                "train/loss_ce":   train_stats["loss_ce"],
                "train/loss_dice": train_stats["loss_dice"],
                "train/loss_edge": train_stats["loss_edge"],
                "train/lr":        lr_now,
                "val/loss":        val_stats["loss_total"],
                "val/mIoU":        val_stats["miou"] * 100,
                "val/BIoU":        val_stats["biou"] * 100,
                "val/pixel_acc":   val_stats["pixel_acc"] * 100,
                "epoch":           epoch,
            }
            # Log per-class IoU (top 20 worst for large-class datasets)
            iou_per_cls = val_stats["iou_per_class"]
            num_classes  = cfg["model"]["num_classes"]
            if num_classes <= 20:
                for c, iou in enumerate(iou_per_cls):
                    wandb_payload[f"val/iou_cls_{c:03d}"] = iou * 100
            else:
                # Log bottom-20 worst-performing classes by name index
                worst_idxs = sorted(range(num_classes), key=lambda i: iou_per_cls[i])[:20]
                for i in worst_idxs:
                    wandb_payload[f"val/iou_worst_cls_{i:03d}"] = iou_per_cls[i] * 100
                wandb_payload["val/worst20_mean_iou"] = (
                    sum(iou_per_cls[i] for i in worst_idxs) / 20 * 100
                )
            wandb_log(cfg, wandb_payload, step=epoch)

            # ── plots ─────────────────────────────────────────────────
            plot_losses(tracker.history, run_dir, epoch)
            plot_loss_components(tracker.history, run_dir, epoch)

            # ── visualizations ────────────────────────────────────────
            if epoch % vis_interval == 0 or epoch == 1:
                vis_imgs, vis_preds, vis_labels, vis_edges, vis_stems = val_stats["vis"]
                if vis_imgs:
                    grid_path = save_segmentation_vis(
                        torch.cat(vis_imgs),
                        torch.cat(vis_preds),
                        torch.cat(vis_labels),
                        run_dir, epoch,
                        edge_preds=torch.cat(vis_edges) if vis_edges else None,
                        stems=vis_stems,
                        n=num_vis_samples,
                        num_classes=cfg["model"]["num_classes"],
                    )
                    logger.info(
                        f"Saved visualizations: epoch-{epoch:04d}/"
                        f"comparison_grid.png + examples/"
                    )
                    # Upload grid image to W&B
                    wandb_log_image(cfg, "val/seg_grid", grid_path, step=epoch)

            # ── checkpoints ───────────────────────────────────────────
            is_best = val_stats["miou"] > best_miou
            if is_best:
                best_miou  = val_stats["miou"]
                best_epoch = epoch
                save_checkpoint(
                    {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "best_miou": best_miou, "best_epoch": best_epoch},
                    run_dir, "best.pth"
                )
                logger.info(
                    f"✓ New best mIoU={best_miou*100:.2f}% (Epoch {epoch}) — best.pth saved"
                )
                if _wandb_enabled(cfg):
                    wandb.run.summary["best_miou"]  = best_miou * 100
                    wandb.run.summary["best_epoch"] = best_epoch

            # Always save latest
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_miou": best_miou, "best_epoch": best_epoch},
                run_dir, "latest.pth"
            )

            if epoch % save_interval == 0:
                save_checkpoint(
                    {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "best_miou": best_miou, "best_epoch": best_epoch},
                    run_dir, f"epoch{epoch:04d}.pth"
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        status = "INTERRUPTED"
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        status = "FAILED"
        raise
    finally:
        end_time = time.time()
        # Save final checkpoint
        if status == "COMPLETED":
            save_checkpoint(
                {"epoch": tcfg["epochs"], "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_miou": best_miou, "best_epoch": best_epoch},
                run_dir, "final.pth"
            )
        # Save run report
        report_path = save_run_report(
            run_dir=run_dir, cfg=cfg, model=model, tracker=tracker,
            start_time=start_time, end_time=end_time, status=status,
            best_miou=best_miou, best_epoch=best_epoch,
        )
        logger.info(f"Report saved → {report_path}")
        logger.info(
            f"[{status}] Best mIoU: {best_miou*100:.2f}% @ epoch {best_epoch}  |  "
            f"Outputs: {run_dir}"
        )
        wandb_finish(cfg)


if __name__ == "__main__":
    main()
