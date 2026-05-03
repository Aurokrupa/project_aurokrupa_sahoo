"""
ImageClassifier - Custom CNN built using PyTorch.
"""

from torch import nn
from config import num_classes

#Convolution Block
class ConvBlock(nn.Module):
    """
    Double Conv Block:
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout2d

    Conv2d - extract features (edges, texture, patterns)
    BatchNorm2d - Stabilizes training and allows higher learning rate
    ReLU - Adds non linearity
    MaxPool2d - Reduces spatial size, keeps important features
    Dropout2d - Randomly drops channels to prevent overfitting

    Two conv layers per block so the model learns richer features 
    at each spatial scale before reducing size.
    """
    def __init__(self, in_channels, out_channels, dropout=0.25):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout)
        )
    def forward(self, x):
        return self.block(x)

#Custom CNN model
class ImageClassifier(nn.Module):
    """
    Custom CNN built for binary image classification.

    Architecture:
        5 x ConvBlock  (channels: 3->32->64->128->256->512)
        GlobalAvgPool  (512 x 4 x 4 -> 512 x 1 x 1)
        Classifier     (512 -> 256 -> 128 -> num_classes)
    Input  : (B, 3, 128, 128)
    Output : (B, num_classes)
    """
    def __init__(self, num_classes=num_classes):
        super(ImageClassifier, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.25),      # 128 -> 64      # dropout prevents overfitting and 
            ConvBlock(32, 64, dropout=0.25),     # 64  -> 32      # forces model to learn robust features
            ConvBlock(64, 128, dropout=0.25),    # 32  -> 16
            ConvBlock(128, 256, dropout=0.25),   # 16  -> 8
            ConvBlock(256, 512, dropout=0.25))   # 8   -> 4

        # Global Average Pooling - converts each feature map into a single number by averaging all its values
        # reduces parameters, prevents overfitting, translation invariance, better interpretability
        self.gap = nn.AdaptiveAvgPool2d(1)           

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

