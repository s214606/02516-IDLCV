import torch
import torch.nn as nn
import torch.nn.functional as F

class IndividualModel(nn.Module):
    """
    Per-frame CNN that outputs a feature vector [B, C, H, W] for a single frame.
    Designed for 64x64 inputs; uses global avg pool to be robust to size.
    """
    """
    Args:
        out_dim: integer referring to the last layer for fixed feature size
        fully_connected: boolean value for 
    """
    def __init__(self, out_dim: int = 128): 
        super(IndividualModel, self).__init__()
        self.out_dim = out_dim
        self.flow = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2),  # 2x2
            #nn.Conv2d(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
            nn.Conv2d(1024, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU(),#shape: [128 x 2 x 2]
        )
    def forward(self, x):
        x = self.flow(x)
        # print(f"the shape of our output shit is {x.shape}")
        return x
        
class LateFusion(nn.Module):
    """Temporal late fusion with a shared per-frame CNN.

    Expects input shaped either [B, C, T, H, W] or [B, T, C, H, W].
    Returns logits of shape [B, num_classes]. Fusion modes:
      - 'average_pooling': per-frame global avg pool → 256-d features → average over time
      - 'fully_connected': return per-frame feature maps, flatten spatially, concatenate across time
      - num frames corresponds to the number of frames per video
      - num classes
    """
    def __init__(self, num_frames: int, num_classes: int, dropout_rate: float = 0.3, fusion: str = 'fully_connected'):
        super(LateFusion, self).__init__()
        assert fusion in {'average_pooling', 'fully_connected'}, "fusion must be 'average_pooling' or 'fully_connected'"
        self.num_frames = num_frames
        self.fusion = fusion
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.encoder = IndividualModel(out_dim=128)
        self.classifier = nn.LazyLinear(self.num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1) #revise why this works or why also (1,1) works

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept stacked tensor either as [B, T, 3, H, W] or [B, 3, T, H, W]
        if not (torch.is_tensor(x) and x.dim() == 5):
            raise TypeError("LateFusion.forward expects a stacked tensor [B, T, 3, H, W] or [B, 3, T, H, W]")

        # If input is [B, 3, T, H, W], permute to [B, T, 3, H, W]
        if x.shape[1] == 3 and x.shape[2] == self.num_frames:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        B, T, C, H, W = x.shape

        # Fold time into batch → encode per-frame → unfold back        
        batch_stacked_frames = x.view(B * T, C, H, W)

        if self.fusion == 'average_pooling':
            feats = self.encoder(batch_stacked_frames)
            #print(f"The size of feats is {feats.shape}") # (ideally) returns a tensor of size [80, 128, 2, 2]
            pooling = self.pool(feats) # Converts into tensor of size [80, 128, 1, 1] by performing GLOBAL AVERAGE POOLING
            #print(f"The size of POOLING FEATS is {pooling.shape}") # (ideally) returns a tensor of size [80, 128, 1, 1]
            pooled_feats = pooling.reshape(B, T, -1)   # (Ideally) Returns a vector of size [8, 10, 128]
            #print(f"POOLED feats look like {pooled_feats.shape}")
            clip_feat = pooled_feats.mean(dim=1) # IDEALLY RETURNS [8, 128]
            #print(f"The shape before passing to linear pooling layer is {clip_feat.shape}")
            logits = self.classifier(clip_feat)            # [128, num_classes]
            return logits
        else:  # 'fully_connected'
            maps = self.encoder(batch_stacked_frames)              # Encoder returns maps [80, 128, 2, 2] 80 -> B*T
            BT, C2, h, w = maps.shape
            #print(f"The first shape of fully connected is {maps.shape}")
            maps = maps.reshape(B, T, C2, h, w)               # [B, T, 128, h, w]
            flat = maps.flatten(start_dim=2)               # [B, T, 128*h*w]
            #print(f"The FLAT shape of fully connected is {flat.shape}")
            fused = flat.flatten(start_dim=1)              # [B, T*128*h*w] (could also do everything at once)
            #print(f"The SECOND FLAT shape of fully connected is {fused.shape}")
            logits = self.classifier(fused)                # [B, num_classes]
            return logits