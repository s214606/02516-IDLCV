import torch
import torch.nn as nn
import torchvision

class RCNN_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 1. Backbone
        # Use valid weights instead of deprecated 'pretrained'
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
            
        self.out_channels = 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # FIX 2: Input is 2048 (from pooling), Output is 512
            nn.Linear(self.out_channels, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            # FIX 3: Input must match previous Output (512)
            nn.Linear(512, num_classes)
        )
        
        # 3. Bounding Box Regressor
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 4) 
        )

    def forward(self, x):
       
        x = self.features(x)      
        x = self.avgpool(x)       

        class_scores = self.classifier(x)   
        bbox_deltas = self.bbox_regressor(x)
        
        return class_scores, bbox_deltas