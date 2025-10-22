import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoStreamVGG(nn.Module):

    def __init__(self, num_classes=10, num_frames =10):
        super().__init__()
        flowchannels_in = num_frames*2
        # Spatial stream (RGB: 3 channels)
        self.spatial_features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  
        )
        self.spatial_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Temporal stream (Optical flow: 20 channels)
        self.temporal_features = nn.Sequential(
            nn.Conv2d(flowchannels_in, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.temporal_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        rgb, flow = x 
        # Spatial stream
        spatial_out = self.spatial_classifier(self.spatial_features(rgb))
        
        # Temporal stream
        temporal_out = self.temporal_classifier(self.temporal_features(flow))
        
        # Fusion: average softmax probabilities
        spatial_probs = F.softmax(spatial_out, dim=1)
        temporal_probs = F.softmax(temporal_out, dim=1)
        
        return (spatial_probs + temporal_probs) / 2
    
