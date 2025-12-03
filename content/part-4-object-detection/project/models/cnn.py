import torch
import torch.nn as nn
import torchvision
from torchvision.ops import RoIAlign

class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
            
        self.out_channels = 2048
        self.featmap_stride = 32

        self.roi_align = RoIAlign(output_size = (7,7),
                                    spatial_scale=1.0/self.featmap_stride,
                                    sampling_ratio = 2)
        self.flatten_dim = self.out_channels *7*7

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU()
        )

        #Two outputs
        self.class_pred = nn.Linear(1024, num_classes +1)
        self.bbox_pred = nn.Linear(1024, (num_classes +1)*4)
    
    def forward(self, images, proposals):

        feature_map = self.features(images)
        
        if isinstance(proposals, torch.Tensor) and proposals.dim() == 3:
            proposals = list(proposals.unbind(0))
        
        roi_features = self.roi_align(feature_map, proposals)
        
        # D. Flatten & Head
        roi_features = roi_features.flatten(start_dim=1)
        fc_out = self.fc(roi_features)
        
        # E. Predictions
        scores = self.class_pred(fc_out)
        bbox_deltas = self.bbox_pred(fc_out)
        
        return scores, bbox_deltas