from torch.nn import CrossEntropyLoss
from metrics.classification import BaseMetric
import torch.nn as nn

class BaseLoss:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
    
    def compute(self):
        raise NotImplementedError
    

class LossFunction(BaseMetric):
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def reset(self):
        self.loss_sum = 0.0
        self.loss_total = 0.0

    def update(self, predictions, labels):
        loss = self.loss_function(predictions, labels)
        self.loss_sum += loss.item()
        self.loss_total += labels.size(0)
        return loss
    
    def compute(self):
        return self.loss_sum / self.loss_total
    
    import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # we want to reduce dimensions across each mask to get a single scalar per mask!
        dims = tuple(range(1, y_pred.dim())) # remember we have [B,C,H,W] so we compress it as [B,1]
        # add a regularization term to address division by zero 
        # apply the sigmoid since we're talking about CONFIDENCE of y_pred
        conf = torch.sigmoid(y_pred)
        numerator = (2 * (conf * y_true) + 1).mean(dim = dims)
        denominator = ((conf + y_true).mean(dim = dims) + 1)
        dice = numerator / denominator
        loss = 1 - dice.mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        gamma = 2
        pred = torch.sigmoid(y_pred)

        calc = (((1-pred)**gamma)*(y_true*pred))+((1-y_true)*pred)
        loss = calc.mean()
        return loss
    
    
class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        regularization = torch.mean()
        return loss + 0.1*regularization

