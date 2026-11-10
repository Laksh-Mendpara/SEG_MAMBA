"""
Multi-dataset loader for SDBC-Mamba.

Supported datasets (selected via cfg["data"]["dataset"]):
  - oxford_pet   : Oxford-IIIT Pet (binary, trimap annotations)
  - cityscapes   : Cityscapes (supports both official leftImg8bit/gtFine and Kaggle train/val img/label)
  - ade20k       : ADE20K (150 classes, ADEChallengeData2016 format)

All datasets return: (img_tensor [3,H,W], label_tensor [H,W] int64, stem str)
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _base_augment(image, mask, image_size, aug_cfg):
    """Synchronized random augmentation (all datasets)."""
    target = (image_size, image_size)

    # Random resized crop
    if aug_cfg.get("random_resized_crop", True):
        scale = aug_cfg.get("crop_scale", [0.5, 2.0])
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=scale, ratio=(0.75, 1.333)
        )
        image = TF.resized_crop(image, i, j, h, w, target, TF.InterpolationMode.BILINEAR)
        mask  = TF.resized_crop(mask,  i, j, h, w, target, TF.InterpolationMode.NEAREST)
    else:
        image = TF.resize(image, target, TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  target, TF.InterpolationMode.NEAREST)

    # Horizontal flip
    if random.random() < aug_cfg.get("h_flip", 0.5):
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    # Random rotation
    if aug_cfg.get("random_rotation", False):
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST,
                          fill=255)

    # Color jitter (image only)
    if aug_cfg.get("color_jitter_prob", 0.8) > 0:
        cj = aug_cfg.get("color_jitter", [0.4, 0.4, 0.4, 0.1])
        jitter = T.ColorJitter(
            brightness=cj[0], contrast=cj[1], saturation=cj[2], hue=cj[3]
        )
        if random.random() < aug_cfg.get("color_jitter_prob", 0.8):
            image = jitter(image)

    # Random grayscale (image only)
    if random.random() < aug_cfg.get("grayscale_prob", 0.1):
        image = TF.to_grayscale(image, num_output_channels=3)

    return image, mask


def _base_val_transform(image, mask, image_size):
    """Deterministic: resize shorter side, then center crop."""
    target = (image_size, image_size)
    image = TF.resize(image, image_size, TF.InterpolationMode.BILINEAR)
    mask  = TF.resize(mask,  image_size, TF.InterpolationMode.NEAREST)
    image = TF.center_crop(image, target)
    mask  = TF.center_crop(mask,  target)
    return image, mask


def _to_tensor(image, mask_np):
    img_t = TF.to_tensor(image)
    img_t = TF.normalize(img_t, _IMAGENET_MEAN, _IMAGENET_STD)
    lbl_t = torch.from_numpy(mask_np).long()
    return img_t, lbl_t


# ═══════════════════════════════════════════════════════════════════════
# 1. Oxford-IIIT Pet Dataset
# ═══════════════════════════════════════════════════════════════════════

def _remap_trimap(mask_np: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    out = np.empty_like(mask_np, dtype=np.int64)
    out[mask_np == 1] = 1            # foreground pet
    out[mask_np == 2] = 0            # background
    out[mask_np == 3] = ignore_index # border → ignore
    return out


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset (binary: pet=1, background=0, border=ignore).
    Trimap pixel values: 1=pet, 2=background, 3=border.
    """

    def __init__(self, root, split="train", splits_dir=None,
                 image_size=512, augment=True, ignore_index=255, aug_cfg=None):
        self.root       = Path(root)
        self.image_dir  = self.root / "images"
        self.mask_dir   = self.root / "annotations" / "trimaps"
        self.image_size = image_size
        self.augment    = augment
        self.ignore_index = ignore_index
        self.aug_cfg    = aug_cfg or {}

        splits_dir = Path(splits_dir) if splits_dir else self.root / "splits"
        split_file = splits_dir / f"{split}.txt"

        if split_file.exists():
            with open(split_file) as f:
                self.stems = [l.strip() for l in f if l.strip()]
        else:
            all_stems = sorted(
                s.stem for s in self.image_dir.glob("*.jpg")
                if (self.mask_dir / f"{s.stem}.png").exists()
            )
            random.seed(42); random.shuffle(all_stems)
            n_train = int(0.8 * len(all_stems))
            splits_map = {"train": all_stems[:n_train], "val": all_stems[n_train:]}
            splits_dir.mkdir(parents=True, exist_ok=True)
            for sp, stems in splits_map.items():
                (splits_dir / f"{sp}.txt").write_text("\n".join(stems))
            self.stems = splits_map[split]

        if not self.stems:
            raise RuntimeError(f"No images found for split='{split}' in {self.root}")

    def __getitem__(self, idx):
        stem  = self.stems[idx]
        image = Image.open(self.image_dir / f"{stem}.jpg").convert("RGB")
        mask  = Image.open(self.mask_dir  / f"{stem}.png")

        if self.augment:
            image, mask = _base_augment(image, mask, self.image_size, self.aug_cfg)
        else:
            image, mask = _base_val_transform(image, mask, self.image_size)

        mask_np = _remap_trimap(np.array(mask, dtype=np.int64), self.ignore_index)
        return *_to_tensor(image, mask_np), stem

    def __len__(self):
        return len(self.stems)


# ═══════════════════════════════════════════════════════════════════════
# 2. Cityscapes Dataset
# ═══════════════════════════════════════════════════════════════════════

# Official Cityscapes label ID → training ID mapping (30 raw → 19 eval + 255 void)
CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,   8: 1,   9: 255, 10: 255,
    11: 2,  12: 3,  13: 4,  14: 255, 15: 255, 16: 255,
    17: 5,  18: 255,
    19: 6,  20: 7,
    21: 8,  22: 9,
    23: 10,
    24: 11, 25: 12,
    26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18,
    -1: 255,
}

_CS_LUT = np.full(256, 255, dtype=np.int64)
for raw_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
    if 0 <= raw_id < 256:
        _CS_LUT[raw_id] = train_id

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

# Standard Cityscapes 19 colors (RGB)
CITYSCAPES_PALETTE = [
    (128, 64, 128),  # 0: road
    (244, 35, 232),  # 1: sidewalk
    (70, 70, 70),    # 2: building
    (102, 102, 156), # 3: wall
    (190, 153, 153), # 4: fence
    (153, 153, 153), # 5: pole
    (250, 170, 30),  # 6: traffic light
    (220, 220, 0),   # 7: traffic sign
    (107, 142, 35),  # 8: vegetation
    (152, 251, 152), # 9: terrain
    (70, 130, 180),  # 10: sky
    (220, 20, 60),   # 11: person
    (255, 0, 0),     # 12: rider
    (0, 0, 142),     # 13: car
    (0, 0, 70),      # 14: truck
    (0, 60, 100),    # 15: bus
    (0, 80, 100),    # 16: train
    (0, 0, 230),     # 17: motorcycle
    (119, 11, 32),   # 18: bicycle
]


def _get_color_lut():
    lut_cache = Path("/home/b23cs1037/cv_project/data/cityscapes_color_lut.npy")
    if lut_cache.exists():
        return np.load(lut_cache)

    palette = np.array(CITYSCAPES_PALETTE, dtype=np.float32)
    lut = np.full((256, 256, 256), 255, dtype=np.uint8)
    g = np.arange(256, dtype=np.float32)
    b = np.arange(256, dtype=np.float32)
    for r_val in range(256):
        grid_g, grid_b = np.meshgrid(g, b, indexing='ij')
        pts = np.stack([np.full_like(grid_g, r_val), grid_g, grid_b], axis=-1)
        dists = np.linalg.norm(pts[:, :, None, :] - palette[None, None, :, :], axis=-1)
        min_d = dists.min(axis=-1)
        cls = dists.argmin(axis=-1)
        is_void = (np.linalg.norm(pts, axis=-1) < 25) | (min_d > 45)
        cls[is_void] = 255
        lut[r_val] = cls.astype(np.uint8)

    lut_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(lut_cache, lut)
    return lut


class CityscapesDataset(Dataset):
    """
    Cityscapes segmentation dataset.
    Supports two formats:
      1. Official structure: root/leftImg8bit/{split}/{city}/*.png and root/gtFine/{split}/{city}/*_labelIds.png
      2. Kaggle (shuvoalok) structure: root/{split}/img/*.png and root/{split}/label/*.png (RGB color masks)
    """

    def __init__(self, root, split="train", image_size=768,
                 augment=True, ignore_index=255, aug_cfg=None):
        self.root       = Path(root)
        self.image_size = image_size
        self.augment    = augment
        self.ignore_index = ignore_index
        self.aug_cfg    = aug_cfg or {}
        self.samples    = []
        self.is_color_mask = False

        # Format 1: Official structure
        img_root_off  = self.root / "leftImg8bit" / split
        mask_root_off = self.root / "gtFine" / split

        # Format 2: Kaggle structure
        img_root_kag  = self.root / split / "img"
        mask_root_kag = self.root / split / "label"

        if img_root_off.exists():
            for city_dir in sorted(img_root_off.iterdir()):
                if not city_dir.is_dir():
                    continue
                for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                    stem = img_path.name.replace("_leftImg8bit.png", "")
                    mask_path = mask_root_off / city_dir.name / f"{stem}_gtFine_labelIds.png"
                    if mask_path.exists():
                        self.samples.append((img_path, mask_path, stem))
        elif img_root_kag.exists():
            self.is_color_mask = True
            self.color_lut = _get_color_lut()
            for img_path in sorted(img_root_kag.glob("*.png")):
                mask_path = mask_root_kag / img_path.name
                if mask_path.exists():
                    self.samples.append((img_path, mask_path, img_path.stem))
        else:
            raise RuntimeError(
                f"Cityscapes directory structure not found under {self.root}.\n"
                f"Expected either leftImg8bit/{split}/ or {split}/img/"
            )

        if not self.samples:
            raise RuntimeError(f"No Cityscapes samples found for split='{split}' under {self.root}")

    def __getitem__(self, idx):
        img_path, mask_path, stem = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)

        if self.augment:
            image, mask = _base_augment(image, mask, self.image_size, self.aug_cfg)
        else:
            image, mask = _base_val_transform(image, mask, self.image_size)

        mask_np = np.array(mask)
        if self.is_color_mask:
            # Map RGB color mask -> class index using precomputed 3D LUT
            if mask_np.ndim == 3 and mask_np.shape[2] >= 3:
                r = mask_np[:, :, 0]
                g = mask_np[:, :, 1]
                b = mask_np[:, :, 2]
                label = self.color_lut[r, g, b].astype(np.int64)
            else:
                label = mask_np.astype(np.int64)
        else:
            # Remap raw labelIds -> trainIds via 1D LUT
            label = _CS_LUT[mask_np.astype(np.uint8)].astype(np.int64)

        return *_to_tensor(image, label), stem

    def __len__(self):
        return len(self.samples)


# ═══════════════════════════════════════════════════════════════════════
# 3. ADE20K Dataset
# ═══════════════════════════════════════════════════════════════════════

ADE20K_CLASS_NAMES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth",
    "door", "table", "mountain", "plant", "curtain", "chair",
]


class ADE20KDataset(Dataset):
    """
    ADE20K semantic segmentation dataset (ADEChallengeData2016 format).

    Directory layout expected:
        root/
          images/training/*.jpg         (20,210 images)
          images/validation/*.jpg       (2,000 images)
          annotations/training/*.png
          annotations/validation/*.png
        (also supports root/ADEChallengeData2016/... automatically)
    """

    _SPLIT_MAP = {"train": "training", "val": "validation"}

    def __init__(self, root, split="train", image_size=512,
                 augment=True, ignore_index=255, aug_cfg=None):
        self.root       = Path(root)
        self.image_size = image_size
        self.augment    = augment
        self.ignore_index = ignore_index
        self.aug_cfg    = aug_cfg or {}

        # Handle nested ADEChallengeData2016 if present
        base_dir = self.root / "ADEChallengeData2016" if (self.root / "ADEChallengeData2016").exists() else self.root

        folder = self._SPLIT_MAP.get(split, split)
        self.img_dir  = base_dir / "images"      / folder
        self.mask_dir = base_dir / "annotations" / folder

        if not self.img_dir.exists():
            raise RuntimeError(
                f"ADE20K image directory not found: {self.img_dir}\n"
                "Expected: root/images/{training|validation}/*.jpg"
            )

        # Match jpg images to their png annotations
        self.samples = []
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            mask_path = self.mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                self.samples.append((img_path, mask_path, img_path.stem))

        if not self.samples:
            raise RuntimeError(
                f"No ADE20K samples found for split='{split}' under {self.root}"
            )

    def __getitem__(self, idx):
        img_path, mask_path, stem = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")  # single-channel uint8

        if self.augment:
            image, mask = _base_augment(image, mask, self.image_size, self.aug_cfg)
        else:
            image, mask = _base_val_transform(image, mask, self.image_size)

        # Remap: 1–150 → 0–149, pixel 0 (unlabelled) → ignore_index
        mask_np = np.array(mask, dtype=np.int64)
        label   = mask_np - 1                        # 1–150 → 0–149
        label[mask_np == 0] = self.ignore_index       # 0 → 255

        return *_to_tensor(image, label), stem

    def __len__(self):
        return len(self.samples)


# ═══════════════════════════════════════════════════════════════════════
# Factory — dispatches on cfg["data"]["dataset"]
# ═══════════════════════════════════════════════════════════════════════

def build_dataloaders(cfg: dict):
    data_cfg = cfg["data"]
    aug_cfg  = cfg["training"].get("augmentation", {})
    bs       = cfg["training"]["batch_size"]
    nw       = data_cfg.get("num_workers", 4)
    dataset_name = data_cfg.get("dataset", "oxford_pet").lower()
    root     = data_cfg["root"]
    img_size = data_cfg["image_size"]
    ign_idx  = data_cfg.get("ignore_index", 255)
    splits_dir = data_cfg.get("splits_dir", None)

    if dataset_name == "oxford_pet":
        train_ds = OxfordPetDataset(
            root=root, split="train", splits_dir=splits_dir,
            image_size=img_size, augment=True,
            ignore_index=ign_idx, aug_cfg=aug_cfg,
        )
        val_ds = OxfordPetDataset(
            root=root, split="val", splits_dir=splits_dir,
            image_size=img_size, augment=False,
            ignore_index=ign_idx,
        )

    elif dataset_name == "cityscapes":
        train_ds = CityscapesDataset(
            root=root, split="train",
            image_size=img_size, augment=True,
            ignore_index=ign_idx, aug_cfg=aug_cfg,
        )
        val_ds = CityscapesDataset(
            root=root, split="val",
            image_size=img_size, augment=False,
            ignore_index=ign_idx,
        )

    elif dataset_name == "ade20k":
        train_ds = ADE20KDataset(
            root=root, split="train",
            image_size=img_size, augment=True,
            ignore_index=ign_idx, aug_cfg=aug_cfg,
        )
        val_ds = ADE20KDataset(
            root=root, split="val",
            image_size=img_size, augment=False,
            ignore_index=ign_idx,
        )

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Choose from: oxford_pet | cityscapes | ade20k"
        )

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True, drop_last=False,
        persistent_workers=nw > 0,
    )
    return train_loader, val_loader
