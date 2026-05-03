"""
This script is to train the model - ImageClassifier
"""

import os
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ImageClassifier
from config import ( epochs, learning_rate, weight_decay, es_patience, 
            lr_patience, lr_factor, model_weights, log_path, results_dir, class_names )
from dataset import get_dataloader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Logging
def log(msg):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    def __init__(self, patience=es_patience):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, validation_loss):
        if self.best_loss is None or validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

#Training and Validation
def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    """
    Runs the full training loop with validation, early stopping, LR scheduling and saves the weights.
    """
    device = next(model.parameters()).device
    validation_loader = get_dataloader("validation")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                mode="min", factor=lr_factor, patience=lr_patience)
    early_stop = EarlyStopping(patience=es_patience)

    train_losses, validation_losses = [], []
    train_accs, validation_accs = [], []
    best_validation_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(images)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
            correct += (out.argmax(1)==labels).sum().item()
            total += labels.size(0)

        tr_loss, tr_acc = loss_sum/total, correct/total

        # Validation
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss = loss_fn(out, labels)

                loss_sum += loss.item() * images.size(0)
                correct += (out.argmax(1)==labels).sum().item()
                total += labels.size(0)

        vl_loss, vl_acc = loss_sum/total, correct/total

        train_losses.append(tr_loss); validation_losses.append(vl_loss)
        train_accs.append(tr_acc); validation_accs.append(vl_acc)

        scheduler.step(vl_loss)

        log(f"Epoch [{epoch+1:02d}/{num_epochs}] "
            f"Train Loss: {tr_loss:.4f} Train Accuracy: {tr_acc:.4f} | "
            f"Validation Loss: {vl_loss:.4f}  Validation Accuracy: {vl_acc:.4f}")
        
        if vl_acc > best_validation_acc:
            best_validation_acc = vl_acc
            torch.save(model.state_dict(), model_weights)
            log(f" --> Saved best weights (val acc: {best_validation_acc:.4f})")
 
        early_stop(vl_loss)
        if early_stop.stop:
            log(f"Early stopping at epoch {epoch+1}")
            break

# Plots
    for values, ylabel, fname in [([train_losses, validation_losses], "Loss", "loss_curve.png"),
            ([train_accs, validation_accs], "Accuracy", "accuracy_curve.png"),]:
        plt.figure(figsize=(8, 5))
        plt.plot(values[0], label=f"Train {ylabel}")
        plt.plot(values[1], label=f"Val {ylabel}")
        plt.xlabel("Epoch"); plt.ylabel(ylabel)
        plt.title(f"{ylabel} Curve"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, fname), dpi=300)
        plt.close()
    
    log(f"\nTraining complete. Best val accuracy: {best_validation_acc:.4f}")

    # Loading weights before returning
    model.load_state_dict(torch.load(model_weights))
    return model

if __name__ == "__main__":
    open(log_path, "w").close()
    log("=" * 60)
    log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device : {device}")
 
    model = ImageClassifier().to(device)
    train_loader = get_dataloader("train")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
 
    train_model(model, epochs, train_loader, loss_fn, optimizer)
