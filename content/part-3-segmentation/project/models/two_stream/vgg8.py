import torch
import torch.nn as nn
import torchvision.models as models

class SpatialStreamVGG(nn.Module):
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

class TemporalStreamVGG(nn.Module):
    def __init__(self, num_classes=10, num_frames=9):
        super().__init__()
        
        
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        flow_channels = num_frames * 2
        

        self.features = nn.Sequential(*list(vgg19.features.children()))
        self.features[0] = nn.Conv2d(flow_channels, 64, 3, padding=1)
        
       
        for name, param in self.features.named_parameters():
            # Freeze first 2 conv blocks, train rest
            if 'features.0.' in name or 'features.7.' in name:  
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(0.9),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
           
        )
    
    def forward(self, flow):
        x = self.features(flow)
        logits = self.classifier(x)
        return logits