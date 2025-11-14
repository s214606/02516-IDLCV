import numpy as np

def dice_overlap(y_pred, y_true):
    """Dice Coefficient (F1 Score for segmentation)."""
    y_pred = np.asarray(y_pred).astype(bool)
    y_true = np.asarray(y_true).astype(bool)
    
    intersection = np.logical_and(y_pred, y_true).sum()
    return 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-8)


def intersection_over_union(y_pred, y_true):
    """Intersection over Union (IoU) / Jaccard Index."""
    y_pred = np.asarray(y_pred).astype(bool)
    y_true = np.asarray(y_true).astype(bool)
    
    intersection = np.logical_and(y_pred, y_true).sum()
    union = np.logical_or(y_pred, y_true).sum()
    return intersection / (union + 1e-8)


def accuracy(y_pred, y_true):
    """Pixel-wise Accuracy."""
    y_pred = np.asarray(y_pred).astype(bool)
    y_true = np.asarray(y_true).astype(bool)
    
    correct = (y_pred == y_true).sum()
    total = y_true.size
    return correct / (total + 1e-8)


def sensitivity(y_pred, y_true):
    """Sensitivity / Recall / True Positive Rate."""
    y_pred = np.asarray(y_pred).astype(bool)
    y_true = np.asarray(y_true).astype(bool)
    
    tp = np.logical_and(y_pred, y_true).sum()
    fn = np.logical_and(~y_pred, y_true).sum()
    return tp / (tp + fn + 1e-8)


def specificity(y_pred, y_true):
    """Specificity / True Negative Rate."""
    y_pred = np.asarray(y_pred).astype(bool)
    y_true = np.asarray(y_true).astype(bool)
    
    tn = np.logical_and(~y_pred, ~y_true).sum()
    fp = np.logical_and(y_pred, ~y_true).sum()
    return tn / (tn + fp + 1e-8)

class BaseMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, predictions, labels):
        raise NotImplementedError
    
    def compute(self):
        raise NotImplementedError

class Accuracy(BaseMetric):
    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions, labels, threshold=0.5):
        """
        predictions: [B, 1, H, W], sigmoid output (0–1)
        labels: [B, 1, H, W], binary 0/1 mask
        """
        preds = (predictions > threshold).long()
        labels = labels.long()

        self.correct += (preds == labels).sum().item()
        self.total += labels.numel()

    def compute(self):
        return self.correct / self.total * 100.0 if self.total > 0 else 0.0

class DiceScore(BaseMetric):
    def reset(self):
        self.intersection = 0
        self.pred_sum = 0
        self.label_sum = 0

    def update(self, predictions, labels, threshold=0.5):
        """
        predictions: [B, 1, H, W], sigmoid output (0–1)
        labels: [B, 1, H, W], binary 0/1 mask
        """
        preds = (predictions > threshold).long()
        labels = labels.long()

        self.intersection += (preds * labels).sum().item()
        self.pred_sum += preds.sum().item()
        self.label_sum += labels.sum().item()

    def compute(self):
        denominator = self.pred_sum + self.label_sum
        if denominator > 0:
            return (2 * self.intersection / denominator) * 100.0
        return 0.0


class IoU(BaseMetric):
    def reset(self):
        self.intersection = 0
        self.union = 0

    def update(self, predictions, labels, threshold=0.5):
        """
        predictions: [B, 1, H, W], sigmoid output (0–1)
        labels: [B, 1, H, W], binary 0/1 mask
        """
        preds = (predictions > threshold).long()
        labels = labels.long()

        self.intersection += (preds * labels).sum().item()
        self.union += (preds + labels).clamp(0, 1).sum().item()

    def compute(self):
        if self.union > 0:
            return (self.intersection / self.union) * 100.0
        return 0.0