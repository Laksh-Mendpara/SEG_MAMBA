"""
Standalone evaluation script.

Usage:
    python evaluate.py --checkpoint /DATA1/b23cs1037/cv/<run>/checkpoints/best.pth \
                       --config    /DATA1/b23cs1037/cv/<run>/config.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.cuda.amp as amp

sys.path.insert(0, str(Path(__file__).parent))
from src.utils   import load_config, save_segmentation_vis
from src.dataset import build_dataloaders
from src.decoder import build_model
from src.losses  import build_loss
from src.metrics import SegmentationMetrics, BoundaryIoUMetric


@torch.no_grad()
def evaluate(args):
    cfg    = load_config(args.config)
    device = torch.device(args.device)

    _, val_loader = build_dataloaders(cfg)
    model         = build_model(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    criterion    = build_loss(cfg).to(device)
    num_classes  = cfg["model"]["num_classes"]
    ignore_index = cfg["data"].get("ignore_index", 255)

    seg_m  = SegmentationMetrics(num_classes, ignore_index)
    biou_m = BoundaryIoUMetric(num_classes, ignore_index)
    total_loss = 0.0

    vis_imgs, vis_preds, vis_lbls, vis_edges, vis_stems = [], [], [], [], []
    num_vis = 8

    for images, labels, stems in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            logits, edge_pred = model(images)
            losses = criterion(logits, edge_pred, labels)

        total_loss += losses["total"].item()
        preds = logits.argmax(1)
        seg_m.update(preds, labels)
        biou_m.update(preds, labels)

        if len(vis_imgs) < num_vis:
            rem = num_vis - len(vis_imgs)
            vis_imgs.append(images[:rem].cpu())
            vis_preds.append(preds[:rem].cpu())
            vis_lbls.append(labels[:rem].cpu())
            if edge_pred is not None:
                vis_edges.append(edge_pred[:rem].cpu())
            vis_stems.extend(stems[:rem])

    res  = seg_m.compute()
    biou = biou_m.compute()

    print(f"\n{'='*50}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Val Loss   : {total_loss / len(val_loader):.4f}")
    print(f"mIoU       : {res['miou']*100:.2f}%")
    print(f"Boundary IoU: {biou*100:.2f}%")
    print(f"Pixel Acc  : {res['pixel_acc']*100:.2f}%")
    print(f"IoU/class  : {[f'{x*100:.1f}%' for x in res['iou_per_class']]}")
    print(f"{'='*50}\n")

    # Save visualization
    run_dir = Path(args.checkpoint).parent.parent
    if vis_imgs:
        import torch as _t
        grid_path = save_segmentation_vis(
            _t.cat(vis_imgs), _t.cat(vis_preds), _t.cat(vis_lbls),
            run_dir, epoch=ckpt.get("epoch", 0),
            edge_preds=_t.cat(vis_edges) if vis_edges else None,
            stems=vis_stems,
            n=num_vis
        )
        print(f"Evaluation visual grid saved to: {grid_path}")
        print(f"Per-example outputs saved to: {grid_path.parent / 'examples'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default=None)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.config is None:
        # Try to find config.yaml in the run dir
        run_dir = Path(args.checkpoint).parent.parent
        args.config = str(run_dir / "config.yaml")
        if not Path(args.config).exists():
            args.config = "configs/sdbc_mamba_pet.yaml"

    evaluate(args)

