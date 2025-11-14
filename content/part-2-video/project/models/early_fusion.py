import torch as t
import torch.nn as nn
from torchvision import models

class EarlyFusion(nn.Module):
    def __init__(self, in_size=64, num_classes=10, num_frames=10):
        super().__init__()
        
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features
        
        # Adapter: project C*T channels down to 3 channels
        # This converts concatenated frames to something VGG16 can process
        self.channel_adapter = nn.Conv2d(
            in_channels=3 * num_frames,  # 30 channels
            out_channels=3,               # 3 channels for VGG16
            kernel_size=1,                # 1x1 conv (no spatial mixing)
            bias=True
        )
        
        # Initialize to average across frames
        with t.no_grad():
            # Each output channel averages corresponding input channels
            weight = t.zeros(3, 3 * num_frames, 1, 1)
            for i in range(3):  # For each RGB channel
                weight[i, i::3, 0, 0] = 1.0 / num_frames  # Average every 3rd channel
            self.channel_adapter.weight.copy_(weight)
            self.channel_adapter.bias.zero_()
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape  # (B, 3, 10, 64, 64)
        
        # Early fusion: concatenate frames in channel dimension
        x = x.reshape(B, C * T, H, W)  # (B, 30, 64, 64)
        
        # Adapter: project back to 3 channels
        x = self.channel_adapter(x)  # (B, 3, 64, 64)
        
        # Standard VGG16 forward with pretrained weights
        x = self.features(x)
        logits = self.classifier(x)
        return logits