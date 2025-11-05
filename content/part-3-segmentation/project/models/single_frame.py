import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SingleFrameCNN(nn.Module):
    def __init__(self, num_classes=10, freeze_features=True):
        super().__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features
        

        # Freeze early layers, only train deeper ones
        for i, layer in enumerate(self.features):
            if i < 16:  # Freeze first ~20 layers
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.9),
        nn.Linear(512, num_classes)
    )
    
    def forward(self, rgb):
        x = self.features(rgb)
        logits = self.classifier(x)
        return logits
  