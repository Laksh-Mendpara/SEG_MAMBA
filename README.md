# SDBC-Mamba: Spectrally Decoupled and Boundary-Clamped State Space Models for Efficient Dense Semantic Segmentation

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2Bcu128-orange.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%20A6000%20(48GB)-green.svg)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Official implementation and NeurIPS-format research paper for **SDBC-Mamba**, an efficient semantic segmentation architecture that integrates continuous-discrete State Space Models (SSMs) with high-frequency spectral decoupling and boundary clamping to achieve linear $\mathcal{O}(N)$ computational complexity without boundary blur.

---

## Authors & Affiliation
* **Laksh Mendpara** (`b23cs1037@iitj.ac.in`) — Department of Computer Science & Engineering, IIT Jodhpur
* **Anshuman Parida** (`b23es1008@iitj.ac.in`) — Department of Electrical Engineering, IIT Jodhpur
* **Sahil Kamboj** (`b23cs1061@iitj.ac.in`) — Department of Computer Science & Engineering, IIT Jodhpur

---

## Key Methodological Innovations

1. **Linear $\mathcal{O}(N)$ Selective State Space Scaling**: Replaces quadratic $\mathcal{O}(N^2 C)$ spatial self-attention with 4-way selective visual SSMs, cutting FLOPs by up to $2.8\times$ and eliminating out-of-memory (OOM) failures on ultra-high resolutions ($2048 \times 2048$).
2. **Laplacian Spectral Decoupling**: Solves the exponential low-pass filtering effect of Zero-Order Hold (ZOH) discretization by decomposing multi-scale visual features into a low-frequency semantic component ($X_{\text{low}}$) routed through the SSM, and a high-frequency edge residual ($X_{\text{high}}$) bypassing recurrent updates.
3. **Anisotropic Boundary $\Delta$-Firewall**: Dynamically modulates and clamps discretization step-size $\mathbf{\Delta}_t$ when crossing sharp semantic gradients ($\|\nabla I\|$), clearing accumulated historical memory ($h_t \to \mathbf{0}$) to prevent feature bleeding across adjacent classes.
4. **Stride-8 Scanning Topology**: Unifies hierarchical encoder features at stride 8 ($H/8 \times W/8$) rather than stride 4, slashing scan steps by $75\%$ ($N_8 = N_4 / 4$) and reducing GPU memory transposition overhead.
5. **Present-Class Filtered Soft Dice + Morphological Edge Supervision**: Formulates multi-class Dice strictly over classes present in the batch targets, eliminating denominator collapse and NaN losses on large label spaces (e.g., ADE20K 150 classes).

---

## Benchmark Performance Highlights

| Dataset | Backbone | Resolution | Params | Best Val mIoU | Best Val BIoU | Pixel Acc | Efficiency & SOTA Highlights |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Oxford-IIIT Pet** | Swin-T | $512 \times 512$ | $28.2\text{M}$ | **$98.47\%$** | **$29.72\%$** | **$99.24\%$** | $+6.17\%$ over DeepLabV3-ResNet50 ($92.30\%$), $+4.62\%$ BIoU over DASA ($25.10\%$). |
| **Cityscapes** | Swin-B (22K) | $768 \times 768$ | $89.7\text{M}$ | **$35.14\%^*$** | **$28.08\%$** | **$94.61\%$** | $16.5\text{ FPS}$ ($2.1\times$ faster than SegFormer-B2); stable 45-epoch run in BFloat16. |
| **ADE20K** | Swin-B (22K) | $512 \times 512$ | $89.7\text{M}$ | **$51.35\%$** | **$36.76\%$** | **$83.07\%$** | Surpasses Swin-B ($48.1\%$), VMamba-S ($50.0\%$), and Focal-B ($49.0\%$) by $+3.25\%$ mIoU. |

*\*Note: Cityscapes mIoU reflects downsampled Kaggle annotations ($256 \times 96$ RGB) interpolated to $768 \times 768$.*

---

## Repository Structure

```
cv_project/
├── configs/                             # YAML Experiment Configurations
│   ├── sdbc_mamba_pet.yaml              # Oxford-IIIT Pet (Swin-T, 512x512, 45 epochs)
│   ├── sdbc_mamba_cityscapes.yaml       # Cityscapes (Swin-B, 768x768, 45 epochs)
│   └── sdbc_mamba_ade20k.yaml           # ADE20K (Swin-B, 512x512, 45 epochs)
├── src/                                 # Modular Source Code
│   ├── backbone.py                      # Hierarchical Swin-T / Swin-B Feature Extractors
│   ├── ssm_block.py                     # Selective 4-Way SS2D + Boundary Clamping
│   ├── decoder.py                       # Unified Stride-8 SDBC-Mamba Architecture
│   ├── dataset.py                       # Dataloaders for Pet, Cityscapes & ADE20K
│   ├── losses.py                        # CE + Present-Filtered Dice + Edge BCE Loss
│   ├── metrics.py                       # Batched mIoU, Pixel Accuracy & Fast BIoU
│   └── utils.py                         # Multi-dataset Visualizations, Palettes & Reports
├── report/                              # NeurIPS Publication Manuscript
│   ├── SDBC_Mamba_NeurIPS_Report.pdf    # Compiled 14-Page Publication PDF
│   └── Neurips Styles/                  # LaTeX Source Files & High-Res Figures
│       ├── main.tex                     # Master Document
│       ├── references.bib               # Bibliography
│       └── figures/                     # Architecture, Teaser, Curves & Visual Grids
├── run_final.sh                         # Launch Script: Oxford-IIIT Pet Training
├── run_cityscapes.sh                    # Launch Script: Cityscapes Training
├── run_ade20k.sh                        # Launch Script: ADE20K Training
├── train.py                             # Master Training Script
├── evaluate.py                          # Offline Evaluation & Metric Reproduction Script
└── requirements.txt                     # Python Package Dependencies
```

---

## Installation & Setup

### 1. Prerequisites
* Linux (Ubuntu 20.04 / 22.04 LTS recommended)
* NVIDIA GPU with CUDA support (CUDA 12.0+)
* Python 3.10, 3.11, or 3.12

### 2. Virtual Environment Setup
```bash
# Clone repository and enter directory
cd /home/b23cs1037/cv_project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Core Dependencies (`requirements.txt`)
```text
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12
numpy>=1.24.0
scipy>=1.10.0
Pillow>=9.5.0
matplotlib>=3.7.0
pyyaml>=6.0
wandb>=0.16.0
kaggle>=1.5.16
```

---

## Dataset Acquisition and Preparation

All datasets should reside under `/DATA1/b23cs1037/cv/data/` (or update `root` in the config YAMLs).

### 1. Oxford-IIIT Pet Dataset
```bash
mkdir -p /DATA1/b23cs1037/cv/data/oxford_pet
cd /DATA1/b23cs1037/cv/data/oxford_pet

# Download images and annotations
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# Extract archives
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz

# Expected directory structure:
# oxford_pet/
# ├── images/*.jpg             (7,349 images)
# ├── annotations/trimaps/*.png(Ground-truth trimaps: 1=FG, 2=BG, 3=Boundary)
# └── splits/                  (train.txt: 3,680, val.txt: 3,669)
```

### 2. Cityscapes Dataset (Kaggle or Official)
Setup your Kaggle API key at `~/.kaggle/kaggle.json` (`chmod 600 ~/.kaggle/kaggle.json`):
```bash
mkdir -p /DATA1/b23cs1037/cv/data/cityscapes
cd /DATA1/b23cs1037/cv/data/cityscapes

# Download via Kaggle CLI
kaggle datasets download -d shuvoalok/cityscapes
unzip -q cityscapes.zip

# Expected directory structure:
# cityscapes/
# ├── train/
# │   ├── img/*.png            (2,975 images)
# │   └── label/*.png          (2,975 color masks)
# └── val/
#     ├── img/*.png            (500 images)
#     └── label/*.png          (500 color masks)
# (Official leftImg8bit/ and gtFine/ structures are also supported automatically)
```

### 3. ADE20K Dataset (ADEChallengeData2016)
```bash
mkdir -p /DATA1/b23cs1037/cv/data/ade20k
cd /DATA1/b23cs1037/cv/data/ade20k

# Download via Kaggle CLI
kaggle datasets download -d awsaf49/ade20k-dataset
unzip -q ade20k-dataset.zip

# Expected directory structure:
# ade20k/ADEChallengeData2016/
# ├── images/
# │   ├── training/*.jpg       (20,210 images)
# │   └── validation/*.jpg     (2,000 images)
# └── annotations/
#     ├── training/*.png       (Raw IDs 1-150, 0=unlabelled)
#     └── validation/*.png     (Raw IDs 1-150, 0=unlabelled)
```

---

## Training Instructions

Launch scripts automatically configure GPU allocation, mixed-precision BFloat16 execution, and W&B logging.

```bash
# 1. Train SDBC-Mamba on Oxford-IIIT Pet (45 epochs, Swin-T, 512x512)
bash run_final.sh

# 2. Train SDBC-Mamba on Cityscapes (45 epochs, Swin-B, 768x768)
bash run_cityscapes.sh

# 3. Train SDBC-Mamba on ADE20K (45 epochs, Swin-B, 512x512)
bash run_ade20k.sh
```

### GPU Selection
By default, the scripts target GPU 1. To target a different GPU:
```bash
CUDA_VISIBLE_DEVICES=0 bash run_final.sh
```

### Custom Configuration Training
You can invoke `train.py` directly with custom configurations:
```bash
python train.py \
    --config configs/sdbc_mamba_ade20k.yaml \
    --exp sdbc_mamba_ade20k_custom \
    --device cuda:0
```

---

## Evaluation & Checkpoint Inspection

To evaluate a trained checkpoint on the validation set and generate metrics:
```bash
python evaluate.py \
    --config configs/sdbc_mamba_pet.yaml \
    --checkpoint /DATA1/b23cs1037/cv/20261109-232441_sdbc_mamba_final/checkpoints/best.pth \
    --output_dir /DATA1/b23cs1037/cv/eval_pet
```

### Run Output Structure
For each experiment, outputs are saved under `/DATA1/b23cs1037/cv/<timestamp>_<exp_name>/`:
* `checkpoints/`: `best.pth`, `epoch*.pth`, and `final.pth`.
* `plots/`: `curves_latest.png` (Loss, mIoU, BIoU) and `loss_components_latest.png` (CE, Dice, Edge BCE).
* `visualizations/`: Evaluation grids for each interval (`comparison_grid.png`) and per-sample folders (`examples/example-*/`).
* `report.json`: Automatically generated summary containing runtime duration, model parameter counts, dataset summary, and best performance metrics.
* `train.log`: Complete timestamped execution log.

---

## Compiling the NeurIPS Research Paper

The manuscript is written in LaTeX conforming to NeurIPS 2025 formatting standards.

```bash
cd "/home/b23cs1037/cv_project/report/Neurips Styles"

# Compile PDF using pdflatex and bibtex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# View output
# main.pdf (also copied to SDBC_Mamba_NeurIPS_Report.pdf in project root)
```

---

## Citation

If you find this work or code helpful in your research, please cite:
```bibtex
@article{mendpara2025sdbc,
  title={SDBC-Mamba: Spectrally Decoupled and Boundary-Clamped State Space Models for Efficient Dense Semantic Segmentation},
  author={Mendpara, Laksh and Parida, Anshuman and Kamboj, Sahil},
  journal={Department of Computer Science and Engineering, Indian Institute of Technology Jodhpur},
  year={2025}
}
```

---

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
