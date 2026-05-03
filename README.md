# ImageClassifier
A binary image classification project that detects whether an image is **AI Generated** or **Real**, built using PyTorch as part of course - (DS3273) Image and Video Processing with Deep Learning.

## Directory Structure
```
project_aurokrupa_sahoo/
├── checkpoints/              # Saved model weights
    └── final_weights.pth

├── data/                     # Sample images for testing
    ├── img01.jpg             # AI-generated images (1–10)             
    └── img20.jpg             # Real images (11–20)

├── output/                   # Results generated after training
    ├── training.log          # Training logs (epoch-wise)
    ├── loss_curve.png        # Loss vs epochs
    ├── accuracy_curve.png    # Accuracy vs epochs
    ├── confusion_matrix.png  # Model performance on test set
    ├── gradcam_samples.png   # Model interpretability (Grad-CAM)
    └── metrics.txt           # Accuracy, Precision, Recall, F1-score

├── config.py                 # Hyperparameters and paths
├── model.py                  # CNN architecture
├── dataset.py                # Data loading and preprocessing
├── train.py                  # Model training script
├── evaluate.py               # Evaluation + Grad-CAM
├── predict.py                # Run predictions on the test images
├── interface.py              # Unified interface for usage/testing
└──dataset/
    ├── train/
    ├── validation/
    └── test/
```
---
## Dataset
**Source:** [Kaggle — AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)
 
| Split | Ratio | AI Generated | Real Images |
|---|---|---|---|
| Train | 85% | ~457 | ~369 |
| Validation | 12% | ~64 | ~52 |
| Test | ~3% | 17 | 14 |
 
The `data/` folder contains 20 sample images (10 per class) from the test split for instructor evaluation 

---

## Model Architecture
 
`ImageClassifier` - custom CNN built from scratch, no pretrained weights.
 
```
Input (B, 3, 128, 128)
    ↓
ConvBlock(3   → 32)    128 × 128 → 64 × 64
ConvBlock(32  → 64)    64 × 64   → 32 × 32
ConvBlock(64  → 128)   32 × 32   → 16 × 16
ConvBlock(128 → 256)   16 × 16   → 8 × 8
ConvBlock(256 → 512)   8 × 8     → 4 × 4
    ↓
Global Average Pooling → (B, 512)
    ↓
Linear(512→256) → ReLU → Linear(256→128) → ReLU → Dropout(0.5)
Linear(128→64)  → ReLU → Dropout(0.3)   → Linear(64→2)
    ↓
Output (B, 2)
```
 
Each ConvBlock: `Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout2d`
 
Global Average Pooling is used instead of flattening - reduces parameters, prevents overfitting and improves translation invariance.
 
---

## Configuration

All hyperparameters are defined in `config.py`:

| Variable | Value | Description |
|---|---|---|
| `resize_x` | 128 | Image width after resizing |
| `resize_y` | 128 | Image height after resizing |
| `input_channel` | 3 | RGB channels |
| `mean` | [0.5, 0.5, 0.5] | Normalization mean -> shifts to [-1, 1] |
| `std` | [0.5, 0.5, 0.5] | Normalization std |
| `batch_size` | 32 | Images per training step |
| `epochs` | 40 | Maximum training epochs |
| `learning_rate` | 0.001 | Initial step size for gradient descent |
| `weight_decay` | 1e-4 | L2 regularization - penalizes large weights |
| `es_patience` | 10 | Early stopping - stops if val loss doesn't improve |
| `lr_patience` | 7 | LR scheduler - reduces LR after N stagnant epochs |
| `lr_factor` | 0.5 | Factor by which LR is reduced |

---
 
**Setup**
 
```bash
pip install -r requirements.txt
```
---
## Results
 
| Metric | Validation | Test |
|---|---|---|
| Accuracy |  |  |
| Precision |  |  |
| Recall |  |  |
| F1 Score |  |  |
---

## File Descriptions

| File | Description |
|---|---|
| `config.py` | All hyperparameters, paths and class names |
| `model.py` | `ImageClassifier` - custom CNN architecture |
| `dataset.py` | `AIRealDataset` class + `get_dataloader()` function |
| `train.py` | `train_model()` - full training loop with early stopping and LR scheduling |
| `evaluate.py` | Evaluation on val + test sets, confusion matrix, Grad-CAM |
| `predict.py` | `predict_images()` - batch inference from file paths |
| `interface.py` | Standardised imports for grading program |
| `split_dataset.sh` | Bash script to split raw dataset into train/validation/test |

---

## Using interface.py

**Retrain:**
```python
import torch
from torch import nn, optim
from interface import TheModel, the_trainer, the_dataloader, total_epochs
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TheModel().to(device)
train_loader = the_dataloader("train")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
model = the_trainer(model, total_epochs, train_loader, loss_fn, optimizer)
```

**Run inference:**
```python
import os
from interface import the_predictor, the_data_dir
 
img_paths = sorted([
    os.path.join(the_data_dir, f)
    for f in os.listdir(the_data_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
results = the_predictor(img_paths)
for r in results:
    print(f"{r['file']} → {r['prediction']} ({r['confidence']}%)")
```