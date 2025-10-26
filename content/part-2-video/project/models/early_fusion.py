import torch as t
import torch.nn as nn
from torchvision import models

class EarlyFusion(nn.Module):
    def __init__(self, in_size=64, num_classes=10, num_frames=10):
        super().__init__()

        in_channels = 3 * num_frames  # 3 channels (RGB) per frame
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        #self.features = vgg16.features

        self.features = nn.Sequential(*list(vgg16.features.children()))
        # Replace first conv layer
        self.features[0] = nn.Conv2d(num_frames * 3, 64, kernel_size=3, padding=1)
        
        # Initialize with averaged pretrained weights
        with t.no_grad():
            pretrained_weight = vgg16.features[0].weight
            new_weight = pretrained_weight.repeat(1, num_frames, 1, 1) / num_frames
            self.features[0].weight.copy_(new_weight)
        
        for i, layer in enumerate(self.features):
            if i <= 9:  # Blocks 1-2 trainable
                for param in layer.parameters():
                    param.requires_grad = True
            else:  # Blocks 3-5 frozen
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self._early_fusion(x)  # B, C*T, H, W
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    def _early_fusion(self, x):
        B, C, T, H, W = x.shape  # (B, 3, 10, 64, 64) ← Changed from T, C, H, W
        x = x.reshape(B, C*T, H, W)  # (B, 30, 64, 64)
        return x