import torch as t
import torch.nn as nn
import torch.nn.functional as F

class FastRCNNLoss(nn.Module):
    def __init__(self, lambda_reg=1.0):
        super().__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.SmoothL1Loss(reduction='none') # Important: none reduction first
        self.lambda_reg = lambda_reg

    def forward(self, cls_scores, bbox_pred, labels, bbox_targets):
        # 1. Classification Loss (Always calculated)
        loss_cls = self.cls_criterion(cls_scores, labels)
        
        # 2. Regression Loss (Only for Positive samples!)
        # In R-CNN, background (label=0) has no bounding box logic.
        
        # Create a mask for positive samples (label > 0)
        pos_mask = (labels > 0).view(-1, 1).expand_as(bbox_pred)
        
        # Calculate regression loss
        loss_reg_raw = self.reg_criterion(bbox_pred, bbox_targets)
        
        # Apply mask: only count loss where label is positive
        loss_reg = (loss_reg_raw * pos_mask.float()).sum() / (pos_mask.float().sum() + 1e-6)
        
        # Total Loss
        total_loss = loss_cls + (self.lambda_reg * loss_reg)
        
        return total_loss

        