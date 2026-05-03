"""
All the hyperparameters and configurations for the model - ImageClassifier
"""

import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(BASE_DIR, "dataset")
sample_data = os.path.join(BASE_DIR, "data")
train_images = os.path.join(images_dir, "train")
validation_images = os.path.join(images_dir, "validation")
test_images = os.path.join(images_dir, "test")

results_dir = "output"
checkpoints_dir = "checkpoints"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

model_weights = os.path.join("checkpoints", "final_weights.pth")
log_path = os.path.join(results_dir, "training.log")

# Image Configuration
resize_x = 128           #Image resized to 128*128
resize_y = 128
input_channel = 3

# Normalizes pixel range [0,1] -> [-1,1]
# Data is centered around 0
# ReLU and BatchNorm work more effectively with zero-centered inputs
# Gradients are more stable
mean = [0.5, 0.5, 0.5]   
std = [0.5, 0.5, 0.5]

# Training configuration
batch_size = 32          #No. of images per training step
epochs = 40           
learning_rate = 0.001    #Step size for gradient descent
weight_decay = 1e-4      #L2 Normalization - Penalises large weights and prevents overfitting       
es_patience = 10         #(early stop patience)Prevents overtraining, stops training if validation doesn't improve for the number of epochs
lr_patience = 7          #Helps fine tuning near minima, if model doesn't improve learning rate is reduced
lr_factor = 0.5          #Factor by which learning rate is reduced

# Classes
class_names = ["AI_Generated_Images", "Real_Images"]
num_classes = len(class_names)
