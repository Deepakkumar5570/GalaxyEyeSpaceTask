# Attention-Based Siamese Late Fusion for EO-SAR Binary Change Detection

---

# Project Title & Description

## Project Title
Attention-Based Siamese Late Fusion for EO-SAR Binary Change Detection

## Description
This project focuses on binary change detection using paired Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite imagery for disaster assessment and remote sensing applications.

The objective is to predict a pixel-wise binary change mask between pre-event and post-event image pairs.

The project evolved from a simple UNet baseline into a research-oriented Attention-Based Siamese Late Fusion architecture inspired by:

* ChangeFormer
* STANet
* DSIFN
* BIT
* Attention-based remote sensing change detection literature

### Final Architecture Components
* Siamese temporal representation learning
* Pretrained ResNet34 encoder
* CBAM attention modules
* Late fusion
* Multi-scale feature fusion
* Dice + Focal loss
* Strong augmentations
* Early stopping
* Multi-GPU training

---


# Requirements

## Python Version

```bash
Python 3.12.6
```

## Dependencies

```txt
torch
torchvision
torchaudio
numpy
opencv-python
matplotlib
tqdm
albumentations
segmentation-models-pytorch
pyyaml
scikit-learn
pandas
Pillow
jupyter
notebook
ipykernel
wandb
tensorboard
einops
timm
opencv-contrib-python

```

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# Environment Setup

## Step 1 — Create Environment

### Conda

```bash
python -m venv venv
```

### Virtual Environment

```bash
venv\scripts\Activate

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# Dataset Structure

## Dataset Splits

| Split | Samples |
|---|---|
| Train | 2781 |
| Validation | 334 |
| Test | 77 |

## Expected Directory Structure

```bash
dataset/
│
├── train/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
│
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
│
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

---

# Training

## Train From Scratch
set dataset path inside 
- configs/
example 
for 
- siamese_unet.py 
- go
```
\GalaxyEyeSpaceTask\configs\siamese_unet.yaml
```
change root path to acording your system 
in my caase
```
ROOT: C:\Users\lenovo\Downloads\GalexyEye
```
then go on
- training part
- do the same for all other models approach


```bash
example
python train.py --config configs/siamese_unet.yaml
```

## Multi-GPU Training

```bash
python -m torch.distributed.launch train.py --config configs/siamese_unet.yaml
```

## Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 256×256 |
| Batch Size | 4 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 50 |
| Early Stopping | Patience = 5 |
| Encoder | ResNet34 |
| Encoder Weights | ImageNet |
| GPUs | 2× Tesla T4 |

---

# Evaluation


## Evaluate on Test Dataset
before evaluation part set path
- go into 

```
evaluate.py


```
# Model Evaluation Guide

This guide explains how to correctly use `evaluate.py` for different trained models.

---

# IMPORTANT RULE

For every model evaluation:

You MUST use:

- correct import
- correct model class
- correct checkpoint weight

All three should belong to the SAME architecture.

Otherwise you will get errors like:

```bash
Missing key(s) in state_dict
Unexpected key(s) in state_dict
RuntimeError while loading model
```

---

# HOW TO USE `evaluate.py`

Open:

```bash
evaluate.py
```

Then modify:

- import statement
- model initialization
- checkpoint path

according to the trained model.

---

# 1. BASIC UNET

## In `evaluate.py`

### Replace Import With

```python
from models.unet.basic_unet import (
    BasicUNet
)
```

### Replace Model Initialization With

```python
model = BasicUNet()
```

### Replace Checkpoint Path With

```python
MODEL_PATH = r"outputs/checkpoints/basic_unet/best_model.pth"
```

### Run

```bash
python evaluate.py --config configs/basic_unet.yaml
```

---

# 2. RESNET34 UNET

## In `evaluate.py`

### Replace Import With

```python
from models.unet.resnet34_unet import (
    ResNet34UNet
)
```

### Replace Model Initialization With

```python
model = ResNet34UNet()
```

### Replace Checkpoint Path With

```python
MODEL_PATH = r"outputs/checkpoints/resnet34_unet/best_model.pth"
```

### Run

```bash
python evaluate.py --config configs/resnet34_unet.yaml
```

---

# 3. ROBUST UNET

## In `evaluate.py`

### Replace Import With

```python
from models.unet.robust_unet import (
    RobustUNet
)
```

### Replace Model Initialization With

```python
model = RobustUNet()
```

### Replace Checkpoint Path With

```python
MODEL_PATH = r"outputs/checkpoints/robust_unet/best_model.pth"
```

### Run

```bash
python evaluate.py --config configs/robust_unet.yaml
```

---

# 4. SIAMESE UNET

## In `evaluate.py`

### Replace Import With

```python
from models.unet.siamese_unet import (
    SiameseUNet
)
```

### Replace Model Initialization With

```python
model = SiameseUNet()
```

### Replace Checkpoint Path With

```python
MODEL_PATH = r"outputs/checkpoints/siamese_unet/best_model.pth"
```

### Run

```bash
python evaluate.py --config configs/siamese_unet.yaml
```

---

# 5. ATTENTION SIAMESE UNET

## In `evaluate.py`

### Replace Import With

```python
from models.attention.attention_siamese_unet import (
    AttentionSiameseUNet
)
```

### Replace Model Initialization With

```python
model = AttentionSiameseUNet()
```

### Replace Checkpoint Path With

```python
MODEL_PATH = r"outputs/attention_siamese_checkpoints/best_attention_model.pth"
```

### Run

```bash
python evaluate.py --config configs/attention_siamese_unet.yaml
```

---

# 6. CHANGEFORMER

## In `evaluate.py`

### Replace Import With

```python
from models.transformers.changeformer import (
    ChangeFormer
)
```

### Replace Model Initialization With

```python
model = ChangeFormer()
```

### Replace Checkpoint Path With

```python
MODEL_PATH = r"outputs/changeformer_checkpoints/best_changeformer_model.pth"
```

### Run

```bash
python evaluate.py --config configs/changeformer.yaml
```

---

# WRONG USAGE ❌

```python
from models.attention.attention_siamese_unet import (
    AttentionSiameseUNet
)

model = AttentionSiameseUNet()

MODEL_PATH = "outputs/checkpoints/siamese_unet/best_model.pth"
```

This is WRONG because:

- attention model expects attention weights
- siamese checkpoint does not contain them

This causes:

```bash
Missing key(s) in state_dict
Unexpected key(s) in state_dict
```

---

# CORRECT USAGE ✅

```python
from models.unet.siamese_unet import (
    SiameseUNet
)

model = SiameseUNet()

MODEL_PATH = "outputs/checkpoints/siamese_unet/best_model.pth"
```

---

# QUICK REFERENCE TABLE

| Model | Import | Checkpoint |
|---|---|---|
| Basic UNet | `BasicUNet` | `basic_unet/best_model.pth` |
| ResNet34 UNet | `ResNet34UNet` | `resnet34_unet/best_model.pth` |
| Robust UNet | `RobustUNet` | `robust_unet/best_model.pth` |
| Siamese UNet | `SiameseUNet` | `siamese_unet/best_model.pth` |
| Attention Siamese UNet | `AttentionSiameseUNet` | `best_attention_model.pth` |
| ChangeFormer | `ChangeFormer` | `best_changeformer_model.pth` |

---

# BEST PRACTICE

Keep separate checkpoint folders for every architecture.

Recommended structure:

```bash
outputs/
│
├── checkpoints/
│   ├── basic_unet/
│   ├── robust_unet/
│   ├── resnet34_unet/
│   └── siamese_unet/
│
├── attention_siamese_checkpoints/
│
└── changeformer_checkpoints/
```

This avoids:
- checkpoint confusion
- architecture mismatch
- loading errors

---

# FINAL NOTE

Whenever you change:

- encoder
- decoder
- fusion method
- attention blocks
- architecture

You MUST train a NEW model.

Old checkpoints cannot be reused with different architectures.
ste path for 
- Weigth
- Root dataset path
for example in my case

```
F:\GalaxyEyeSpaceTask\evaluate.py

```
Paths are 
```
DATASET_PATH = r"C:\Users\lenovo\Downloads\GalexyEye"

MODEL_PATH = r"C:\Users\lenovo\Downloads\best_attention_model.pth"

```
Change According to your setup 
run command

```bash
  python evaluate.py

```

## Inference

```bash
 python evaluate.py
```

---

# Model Weights

## Final Checkpoint

```txt
https://drive.google.com/drive/folders/18XIxK3-ODxrvR331ifcPlmH4Bh6fij0k
```

## Example

```txt
https://drive.google.com/drive/folders/18XIxK3-ODxrvR331ifcPlmH4Bh6fij0k
```

---

# Results

## Final Results on Test Dataset

| Model | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| Basic UNet | 0.2542 | 0.6002 | 0.3489 | 0.3631 |
| Robust UNet | 0.2046 | 0.4599 | 0.2587 | 0.3290 |
| UNet + ResNet34 | 0.2493 | 0.5610 | 0.3226 | 0.3871 |
| Siamese Late Fusion | 0.2680 | 0.6661 | 0.3165 | 0.4172 |
| Attention Siamese Late Fusion | 0.3707 | 0.6679 | 0.4557 | 0.5399 |

---

# Citation / References

## Papers
* ChangeFormer
* STANet
* DSIFN
* BIT
* UNet
* CBAM Attention Paper

## Libraries & Codebases
* Segmentation Models PyTorch
* PyTorch
* Albumentations
* OpenCV

---

# Abstract

This project presents an Attention-Based Siamese Late Fusion framework for EO-SAR binary change detection using paired pre-event and post-event satellite imagery.

The proposed architecture combines:

* Siamese temporal representation learning
* Pretrained ResNet34 encoders
* CBAM attention modules
* Late fusion strategies

to improve localization and generalization performance under multimodal domain shifts.

---

# Literature Survey

## ChangeFormer
Transformer-based change detection using token-level temporal fusion.

## STANet
Spatial-temporal attention mechanism for remote sensing change detection.

## DSIFN
Deep Siamese feature fusion network for change detection.

## BIT
Bitemporal image transformer for temporal reasoning.

## UNet
Encoder-decoder architecture widely used in segmentation tasks.

---

# Methodology

## Data Processing
* Binary label remapping
* Patch extraction
* Invalid region filtering
* Data augmentation
* Normalization

## Architecture
* Siamese ResNet34 encoder
* Shared weights
* CBAM attention modules
* Late fusion
* Multi-scale decoder
* Binary segmentation head

## Loss Function
Combined:
* Dice Loss
* Focal Loss

---

# Results Analysis

## Key Observations
* Early fusion performs poorly for EO-SAR modality combinations.
* Siamese temporal separation improves feature learning.
* Attention-guided fusion significantly improves localization quality.
* Pretrained encoders improve semantic representation learning.
* Severe class imbalance strongly affects IoU and recall.

---

# Future Work

## Potential Improvements
* Transformer temporal attention
* ChangeFormer-style fusion
* Frequency-aware transformers
* Self-supervised EO-SAR pretraining
* Dynamic temporal attention
* Valid-region-aware masking

---

# Conclusion

This work demonstrates that:

* Siamese temporal representation learning
* Attention-guided late fusion

significantly improve EO-SAR binary change detection performance.

The final Attention-Based Siamese Late Fusion architecture achieved:

* Strong unseen-scene generalization
* Improved localization quality
* Better performance compared to baseline UNet approaches

---

# Resource Log

## Training Platform
* Kaggle

## Hardware
* 2× Tesla T4 GPUs
* 15GB VRAM each

## Training Time
* ~35–40 sec/epoch
* ~5–7 hours total experimentation

---

# Repo Structure

```bash
├── configs/
├── galaxeye_dataset/
├── losses/
├── metrics/
├── models/
├── outputs/
├── scripts/
├── requirements.txt
├── README.md
```

---

# Problem Statement

Given a co-registered pre-event and post-event EO-SAR image pair, the model predicts a binary pixel-level change mask:

* 0 → No Change
* 1 → Change

The task focuses on identifying disaster-induced structural changes such as damaged or destroyed buildings.

---

# Dataset Understanding & EDA

## Image Properties

| Property | Value |
|---|---|
| Resolution | 1024×1024 |
| EO Channels | 3 (RGB) |
| SAR Channels | 1 |
| Format | TIFF (.tif) |

---

## Original Mask Classes

| Class | Meaning |
|---|---|
| 0 | Background |
| 1 | Intact |
| 2 | Damaged |
| 3 | Destroyed |

---

## Binary Label Remapping

| Original Label | Binary Label |
|---|---|
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |

---

# Dataset Challenges

| Challenge | Cause |
|---|---|
| Severe Imbalance | Sparse changes |
| Weak Generalization | Domain shift |
| Modality Mismatch | EO vs SAR |
| Noisy SAR | Speckle noise |
| Sparse Foreground | Tiny change regions |
| Invalid Regions | Acquisition artifacts |

---

# Architecture Evolution

| Version | Architecture | Key Idea |
|---|---|---|
| V1 | Basic UNet | Baseline segmentation |
| V2 | Robust Training UNet | Better loss + augmentation |
| V3 | UNet + ResNet34 | Pretrained encoder |
| V4 | Siamese Late Fusion | Temporal separation |
| V5 | Attention Siamese Late Fusion | Attention-guided fusion |

---

# Final Architecture

```text
Pre Image ─► Shared Encoder ─┐
                             │
                             ▼
                     Attention Fusion
                             ▲
                             │
Post Image ─► Shared Encoder ─┘
                             ↓
                          Decoder
                             ↓
                       Binary Mask
```

---

# Visualization Results

The project includes:

* Prediction visualizations
* Ground truth vs prediction comparisons
* Random test inference samples
* Binary mask verification

These visualizations help analyze:

* False positives
* Missed regions
* Localization quality
* Semantic reasoning behavior

---

# Assignment Alignment

This project satisfies assignment requirements by:

* Performing extensive EDA
* Analyzing EO-SAR modality differences
* Justifying architecture choices
* Comparing multiple architectures
* Reporting evaluation metrics
* Providing visualization analysis
* Demonstrating systematic experimentation


# Refrences to my work space 


Github link 
-  for Locally setup Environment
  
                  https://github.com/Deepakkumar5570/GalaxyEyeSpaceTask
   
- For Kaggle enviorment
  
                   https://github.com/Deepakkumar5570/GalaxyEyeSpace

- Kaggle Link

             https://www.kaggle.com/code/deepakkumari2004/galaxytask

---
