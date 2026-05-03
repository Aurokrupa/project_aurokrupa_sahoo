"""
Contains Dataset class and DataLoader for the ImageClassifier.
"""

import os
import subprocess
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from config import (train_images, validation_images, test_images,
    resize_x, resize_y, batch_size,mean, std, class_names)

# Transforms 
train_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

eval_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# Dataset 
class AIRealDataset(Dataset):
    """
    Dataset for AI Generated vs Real image classification.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls:i for i, cls in enumerate(class_names)}

        for cls in class_names:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                raise ValueError(f"Missing class folder:{cls_dir}")
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path)

        # Handle palette images with transparency
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# DataLoader
def get_dataloader(split="train"):
    """
    Returns a DataLoader for the given split.
    Args: split : "train" | "validation" | "test"
    Returns: DataLoader
    """
    dirs = {"train" : (train_images, train_transform),
        "validation": (validation_images, eval_transform),
        "test" : (test_images, eval_transform)}

    assert split in dirs, f"split must be one of {list(dirs.keys())}"
    root, tf = dirs[split]

    # # Remove hidden checkpoint folders created by Jupyter/Colab
    # subprocess.run(
    #     ["find", root, "-name", ".ipynb_checkpoints",
    #      "-type", "d", "-exec", "rm", "-rf", "{}", "+"],
    #     stderr=subprocess.DEVNULL
    # )

    dataset = AIRealDataset(root_dir=root, transform=tf)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=2, pin_memory=True)

    print(f"{split:12s} : {len(dataset)} images | {len(loader)} batches")
    return loader
