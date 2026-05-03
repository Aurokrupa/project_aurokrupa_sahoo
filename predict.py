"""
Batch inference function for the ImageClassifier.

Run:
    python predict.py                                        # all images in data/
    python predict.py --images data/img01.jpg data/img02.jpg # specific images
    python predict.py --images path/to/any_image.jpg         # any new image
    python predict.py --data_dir my_folder/                  # different folder
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import resize_x, resize_y, mean, std, class_names, model_weights, sample_data
from model import ImageClassifier


# Transform
infer_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Inference 
class InferenceDataset(Dataset):
    """Loads images from a list of file paths for inference."""
    def __init__(self, img_paths):
        self.img_paths = img_paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = Image.open(path)
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        return infer_transform(image.convert("RGB")), path

# Dataloader for inference
def inferloader(list_of_img_paths):
    """
    Creates a DataLoader from a list of image paths.
    """
    dataset = InferenceDataset(list_of_img_paths)
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# Load model
_model = None
def _load_model():
    global _model
    if _model is not None:
        return _model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier().to(device)
    assert os.path.exists(model_weights), \
        f"Weights not found at '{model_weights}'. Run train.py first."
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()
    _model = model
    return model

# Batch Inference
def predict_images(list_of_img_paths):
    """
    Runs batch inference on a list of image file paths.
    Returns:
        list of dicts, one per image:
        [{
            "file"       : "img01.jpg",
            "prediction" : "AI_Generated_Images",
            "confidence" : 92.3,
            "scores"     : {"AI_Generated_Images": 92.3, "Real_Images": 7.7}
        }, ...]
    """
    model  = _load_model()
    device = next(model.parameters()).device
    loader  = inferloader(list_of_img_paths)
    results = []
    with torch.no_grad():
        for batch, paths in loader:
            batch  = batch.to(device)

            # Forward pass — entire batch at once
            logits = model(batch)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            for i, path in enumerate(paths):
                pred_idx = int(probs[i].argmax())
                results.append({
                    "file"      : os.path.basename(path),
                    "prediction": class_names[pred_idx],
                    "confidence": round(float(probs[i][pred_idx]) * 100, 2),
                    "scores"    : {cls: round(float(probs[i][j]) * 100, 2)
                                   for j, cls in enumerate(class_names)}
                })
    return results

# Helper to load all the images from a directory
def load_images_from_dir(directory):
    """Returns sorted list of all image paths in a directory."""
    supported = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(supported)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images.")
    parser.add_argument("--images", nargs="+", default=None,
        help="Paths to specific image files to classify.")
    parser.add_argument("--data_dir", type=str, default=sample_data,
        help=f"Directory to load all images from (default: {sample_data})")
    args = parser.parse_args()

    if args.images:
        img_paths = args.images
        print(f"Running inference on {len(img_paths)} specified image(s).")
    else:
        img_paths = load_images_from_dir(args.data_dir)
        print(f"No images specified. Loading all {len(img_paths)} images from '{args.data_dir}'")

    if not img_paths:
        print("No images found. Check the path and try again.")
        exit(1)

    results = predict_images(img_paths)

    print(f"\n{'File':<30} {'Prediction':<25} {'Confidence'}")
    print("=" * 70)
    for r in results:
        print(f"{r['file']:<30} {r['prediction']:<25} {r['confidence']:.1f}%")
        for cls, score in r["scores"].items():
            print(f"  {cls:<28} {score:.1f}%")
        print()

    # Summary
    ai_count = sum(1 for r in results if r["prediction"] == class_names[0])
    real_count = sum(1 for r in results if r["prediction"] == class_names[1])
    print("=" * 70)
    print(f"Total images : {len(results)}")
    print(f"AI Generated : {ai_count}")
    print(f"Real Images  : {real_count}")