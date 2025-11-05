import torch as t
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class C3D(nn.Module):
    def __init__(self, in_size=64, num_classes=10, num_frames=10, pretrained=True):
        super().__init__()
        
        # Load pretrained R3D-18 or create from scratch
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
            base_model = r3d_18(weights=weights)
        else:
            base_model = r3d_18(weights=None)
        
        # Extract feature extractor (everything except avgpool and fc)
        self.features = nn.Sequential(
            base_model.stem,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
        
        # Adaptive pooling to handle 64x64 input and 10 frames
        # Output will be (B, 512, 1, 1, 1) regardless of input size
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Custom classifier matching your original structure
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (B, C, T, H, W) = (B, 3, 10, 64, 64)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
