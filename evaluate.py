"""
Evaluates ImageClassifier on validation and test sets.
Also generates Grad-CAM visualizations.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from model import ImageClassifier
from config import (resize_x, resize_y, mean, std, validation_images, test_images,
    results_dir, model_weights, log_path, batch_size)

# Logging
def log(msg):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")

# DataLoader
def get_loader(directory):
    tf = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    ds = datasets.ImageFolder(directory, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, ds.classes


# Evaluation
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            preds = out.argmax(1)     # picks class with highest score
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds, correct / total

# Confusion Matrix
def save_confusion_matrix(true_labels, pred_labels, class_names, filename):
    cm = confusion_matrix(true_labels, pred_labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {filename.replace('_', ' ').title()}")
    plt.tight_layout()
    path = os.path.join(results_dir, f"{filename}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    log(f"Confusion matrix saved to {path}")


# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, tensor, class_idx=None):
        self.model.eval()
        out = self.model(tensor)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = F.interpolate(cam, size=(resize_x, resize_y),
            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        confidence = F.softmax(out, dim=1)[0][class_idx].item() * 100
        return cam, class_idx, confidence

def overlay(tensor, cam):
    img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * np.array(std) + np.array(mean), 0, 1)
    heatmap = colormap.jet(cam)[:, :, :3]
    blended = np.clip(0.5 * img + 0.5 * heatmap, 0, 1)
    return img, blended

def save_gradcam(model, device, class_names, num_samples=8):
    tf = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    ds = datasets.ImageFolder(test_images, transform=tf)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # Target = last conv in last ConvBlock
    target_layer = model.features[-1].block[-3]
    gradcam = GradCAM(model, target_layer)
    samples = []
    for images, labels in loader:
        samples.append((images.to(device), labels))
        if len(samples) == num_samples:
            break
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
    fig.suptitle("Grad-CAM Visualization — ImageClassifier", fontsize=13, fontweight="bold")
    for ax, title in zip(axes[0], ["Original", "Heatmap", "Overlay"]):
        ax.set_title(title, fontsize=11)

    for i, (images, labels) in enumerate(samples):
        cam, pred_idx, conf = gradcam.generate(images)
        orig, blended = overlay(images.cpu(), cam)
        true_lbl = class_names[labels.item()]
        pred_lbl = class_names[pred_idx]
        color = "green" if true_lbl == pred_lbl else "red"
        axes[i][0].imshow(orig)
        axes[i][0].set_ylabel(
            f"True: {true_lbl}\nPred: {pred_lbl} ({conf:.1f}%)", fontsize=8, color=color)
        axes[i][1].imshow(cam, cmap="jet")
        axes[i][2].imshow(blended)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(results_dir, "gradcam_samples.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"Grad-CAM saved to {path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    log(f"Loaded weights from {model_weights}")

    # Validation
    log("\n" + "=" * 60)
    log("VALIDATION RESULTS")
    log("=" * 60)
    validation_loader, class_names = get_loader(validation_images)
    true_l, pred_l, val_acc = evaluate(model, validation_loader, device)
    log(f"Validation Accuracy : {val_acc:.4f}")
    log("\nClassification Report (Validation):")
    log(classification_report(true_l, pred_l, target_names=class_names))
    save_confusion_matrix(true_l, pred_l, class_names, "validation_confusion_matrix")

    # Test
    log("\n" + "=" * 60)
    log("TEST RESULTS")
    log("=" * 60)
    test_loader, class_names = get_loader(test_images)
    true_l, pred_l, test_acc = evaluate(model, test_loader, device)
    log(f"Test Accuracy : {test_acc:.4f}")
    log("\nClassification Report (Test):")
    log(classification_report(true_l, pred_l, target_names=class_names))
    save_confusion_matrix(true_l, pred_l, class_names, "confusion_matrix")

    # Grad-CAM
    log("\nGenerating Grad-CAM visualizations...")
    save_gradcam(model, device, class_names)

    # Save metrics for compare.py
    p, r, f, _ = precision_recall_fscore_support(true_l, pred_l, average="weighted")
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, "w") as f_out:
        f_out.write(f"model=ImageClassifier\n")
        f_out.write(f"test_accuracy={test_acc:.4f}\n")
        f_out.write(f"precision={p:.4f}\n")
        f_out.write(f"recall={r:.4f}\n")
        f_out.write(f"f1={f:.4f}\n")
    log(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
