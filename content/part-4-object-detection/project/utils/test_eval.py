import torch as t 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms, box_iou
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import settings
from data.dataloaders import test_loader
from models.classifier import RCNN_Classifier


model = RCNN_Classifier(num_classes=2)
checkpoint = t.load('checkpoints/Fast-RCNN, resnet50_last.pth', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(settings.device)
model.eval()


def apply_nms_per_image(boxes, scores, preds, image_names, threshold=0.5, conf_threshold=0.3):
    """Apply NMS separately for each image, only to pothole predictions"""
    all_keep_indices = []
    
    unique_images = list(set(image_names))
    
    for img_name in unique_images:
        img_mask = t.tensor([name == img_name for name in image_names])
        img_indices = t.where(img_mask)[0]
        
        img_boxes = boxes[img_indices]
        img_scores = scores[img_indices]
        img_preds = preds[img_indices]
        
        # Only keep pothole predictions
        pothole_mask = img_preds == 1
        pothole_indices = t.where(pothole_mask)[0]
        
        if len(pothole_indices) == 0:
            continue
            
        pothole_boxes = img_boxes[pothole_indices]
        pothole_scores = img_scores[pothole_indices, 1]

            # Filter by confidence first
        conf_mask = pothole_scores >= conf_threshold
        if conf_mask.sum() == 0:
            continue
        
        pothole_boxes = pothole_boxes[conf_mask]
        pothole_scores = pothole_scores[conf_mask]
        pothole_indices = pothole_indices[conf_mask]
        
        # Apply NMS
        keep_local = nms(pothole_boxes, pothole_scores, threshold)
        
        # Convert back to global indices
        keep_global = img_indices[pothole_indices[keep_local]]
        all_keep_indices.append(keep_global)
    
    if len(all_keep_indices) > 0:
        return t.cat(all_keep_indices)
    else:
        return t.tensor([], dtype=t.long)


def calculate_detection_metrics(pred_boxes, pred_scores, pred_labels, pred_image_names,
                                gt_boxes, gt_labels, gt_image_names, 
                                iou_threshold=0.5):
    """
    Calculate object detection metrics (precision, recall, AP) using IoU matching.
    
    Args:
        pred_boxes: Predicted bounding boxes (N, 4)
        pred_scores: Confidence scores (N, num_classes)
        pred_labels: Predicted class labels (N,)
        pred_image_names: Image name for each prediction (list of N)
        gt_boxes: Ground truth bounding boxes (M, 4)
        gt_labels: Ground truth class labels (M,)
        gt_image_names: Image name for each GT (list of M)
        iou_threshold: IoU threshold for matching predictions to GT
    
    Returns:
        Dictionary with metrics: TP, FP, FN, precision, recall, AP
    """
    
    unique_images = list(set(pred_image_names + gt_image_names))
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    all_scores = []
    all_matches = []  # 1 if TP, 0 if FP
    
    total_gt_potholes = 0
    
    for img_name in unique_images:
        # Get predictions for this image
        pred_mask = t.tensor([name == img_name for name in pred_image_names])
        pred_img_indices = t.where(pred_mask)[0]
        
        # Filter to only pothole predictions (class 1)
        img_pred_labels = pred_labels[pred_img_indices]
        pothole_pred_mask = img_pred_labels == 1
        pothole_pred_indices = pred_img_indices[pothole_pred_mask]
        
        if len(pothole_pred_indices) == 0:
            img_pred_boxes = t.zeros((0, 4))
            img_pred_scores = t.zeros(0)
        else:
            img_pred_boxes = pred_boxes[pothole_pred_indices]
            img_pred_scores = pred_scores[pothole_pred_indices, 1]  # Score for pothole class
        
        # Get ground truth for this image
        gt_mask = t.tensor([name == img_name for name in gt_image_names])
        gt_img_indices = t.where(gt_mask)[0]
        
        # Filter to only pothole ground truth (class 1)
        img_gt_labels = gt_labels[gt_img_indices]
        pothole_gt_mask = img_gt_labels == 1
        pothole_gt_indices = gt_img_indices[pothole_gt_mask]
        
        if len(pothole_gt_indices) == 0:
            img_gt_boxes = t.zeros((0, 4))
        else:
            img_gt_boxes = gt_boxes[pothole_gt_indices]
        
        num_gt = len(img_gt_boxes)
        total_gt_potholes += num_gt
        
        if num_gt == 0 and len(img_pred_boxes) == 0:
            # No predictions, no GT - perfect for this image
            continue
        elif num_gt == 0:
            # No GT but we have predictions - all FP
            fp += len(img_pred_boxes)
            for score in img_pred_scores:
                all_scores.append(score.item())
                all_matches.append(0)  # FP
            continue
        elif len(img_pred_boxes) == 0:
            # GT exists but no predictions - all FN
            fn += num_gt
            continue
        
        # Calculate IoU between all predictions and all GT boxes
        iou_matrix = box_iou(img_pred_boxes, img_gt_boxes)
        
        # Sort predictions by confidence (descending)
        sorted_indices = t.argsort(img_pred_scores, descending=True)
        
        gt_matched = t.zeros(num_gt, dtype=t.bool)
        
        for pred_idx in sorted_indices:
            max_iou, max_gt_idx = iou_matrix[pred_idx].max(dim=0)
            
            score = img_pred_scores[pred_idx].item()
            all_scores.append(score)
            
            if max_iou >= iou_threshold and not gt_matched[max_gt_idx]:
                # True positive
                tp += 1
                gt_matched[max_gt_idx] = True
                all_matches.append(1)  # TP
            else:
                # False positive (either low IoU or GT already matched)
                fp += 1
                all_matches.append(0)  # FP
        
        # Count unmatched GT boxes as false negatives
        fn += (~gt_matched).sum().item()
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate Average Precision (AP)
    ap = calculate_ap(all_scores, all_matches)
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'AP': ap * 100,
        'total_gt': total_gt_potholes
    }


def calculate_ap(scores, matches):
    """
    Calculate Average Precision using 11-point interpolation.
    
    Args:
        scores: List of confidence scores
        matches: List of 1 (TP) or 0 (FP) for each prediction
    
    Returns:
        Average Precision value
    """
    if len(scores) == 0:
        return 0.0
    
    # Sort by confidence score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    matches = np.array(matches)[sorted_indices]
    
    # Calculate precision and recall at each threshold
    tp_cumsum = np.cumsum(matches)
    fp_cumsum = np.cumsum(1 - matches)
    
    recalls = tp_cumsum / tp_cumsum[-1] if tp_cumsum[-1] > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def calculate_accuracy(scores, labels):
    """Calculate simple classification accuracy"""
    _, preds = t.max(scores, 1)
    correct = (preds == labels).sum().item()
    return correct, preds


# Collect all predictions
all_scores = t.tensor([])
all_labels = t.tensor([])
all_preds = t.tensor([])
all_boxes = t.tensor([])
all_images = []

with t.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        print(f"Batch {batch_idx + 1}/{len(test_loader)} processed.")
        image = batch['image']
        label = batch['label']
        bbox_target = batch['bbox_target']
        bbox = batch['bbox']
        image_name = batch['image_name']
        region_id = batch['region_id']

        image = image.to(settings.device)
        label = label.to(settings.device)
        bbox_target = bbox_target.to(settings.device)
        bbox = bbox.to(settings.device)

        # Forward pass
        cls_scores, bbox_deltas_pred = model(image)
        cls_scores = F.softmax(cls_scores, dim=1)
        preds = t.argmax(cls_scores, dim=1)

        # Store all predictions
        all_scores = t.concat((all_scores, cls_scores.cpu()), dim=0)
        all_labels = t.concat((all_labels, label.cpu()), dim=0)
        all_preds = t.concat((all_preds, preds.cpu()), dim=0)
        all_boxes = t.concat((all_boxes, bbox.cpu()), dim=0)
        all_images.extend(image_name)


print("=" * 70)
print("CLASSIFICATION METRICS (Before NMS)")
print("=" * 70)

# Traditional classification metrics
tn_before, fp_before, fn_before, tp_before = confusion_matrix(
    all_labels.cpu(), all_preds.cpu(), labels=[0, 1]
).ravel().tolist()

print(f"Confusion Matrix: TN={tn_before}, FP={fp_before}, FN={fn_before}, TP={tp_before}")

correct_before, _ = calculate_accuracy(all_scores, all_labels)
accuracy_before = correct_before / len(all_labels) * 100
recall_before = tp_before / (tp_before + fn_before) * 100 if (tp_before + fn_before) > 0 else 0
precision_before = tp_before / (tp_before + fp_before) * 100 if (tp_before + fp_before) > 0 else 0
f1_before = 2 * (precision_before * recall_before) / (precision_before + recall_before) if (precision_before + recall_before) > 0 else 0

print(f"Classification Accuracy: {accuracy_before:.2f}%")
print(f"Classification Precision: {precision_before:.2f}%")
print(f"Classification Recall: {recall_before:.2f}%")
print(f"Classification F1 Score: {f1_before:.2f}%")


print("\n" + "=" * 70)
print("OBJECT DETECTION METRICS (Before NMS)")
print("=" * 70)

# Object detection style evaluation - before NMS
metrics_before = calculate_detection_metrics(
    all_boxes, all_scores, all_preds, all_images,
    all_boxes, all_labels, all_images,
    iou_threshold=0.5
)

print(f"TP: {metrics_before['TP']}, FP: {metrics_before['FP']}, FN: {metrics_before['FN']}")
print(f"Total GT Potholes: {metrics_before['total_gt']}")
print(f"Detection Precision: {metrics_before['precision']:.2f}%")
print(f"Detection Recall: {metrics_before['recall']:.2f}%")
print(f"Detection F1 Score: {metrics_before['f1']:.2f}%")
print(f"Average Precision (AP): {metrics_before['AP']:.2f}%")


print("\n" + "=" * 70)
print("APPLYING NMS")
print("=" * 70)

keep_indices = apply_nms_per_image(all_boxes, all_scores, all_preds, all_images, threshold=0.95, conf_threshold=0.15)

print(f"Total predictions: {len(all_preds)}")
print(f"Predictions kept after NMS: {len(keep_indices)}")
print(f"Predictions suppressed: {len(all_preds) - len(keep_indices)}")

# Filter to NMS-kept predictions
all_scores_nms = all_scores[keep_indices]
all_preds_nms = all_preds[keep_indices]
all_boxes_nms = all_boxes[keep_indices]
all_images_nms = [all_images[i] for i in keep_indices.tolist()]


print("\n" + "=" * 70)
print("OBJECT DETECTION METRICS (After NMS)")
print("=" * 70)

# Object detection style evaluation - after NMS
metrics_after = calculate_detection_metrics(
    all_boxes_nms, all_scores_nms, all_preds_nms, all_images_nms,
    all_boxes, all_labels, all_images,  # Compare against ALL ground truth
    iou_threshold=0.5
)

print(f"TP: {metrics_after['TP']}, FP: {metrics_after['FP']}, FN: {metrics_after['FN']}")
print(f"Total GT Potholes: {metrics_after['total_gt']}")
print(f"Detection Precision: {metrics_after['precision']:.2f}%")
print(f"Detection Recall: {metrics_after['recall']:.2f}%")
print(f"Detection F1 Score: {metrics_after['f1']:.2f}%")
print(f"Average Precision (AP): {metrics_after['AP']:.2f}%")


print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Precision: {metrics_before['precision']:.2f}% → {metrics_after['precision']:.2f}% "
      f"({metrics_after['precision'] - metrics_before['precision']:+.2f}%)")
print(f"Recall:    {metrics_before['recall']:.2f}% → {metrics_after['recall']:.2f}% "
      f"({metrics_after['recall'] - metrics_before['recall']:+.2f}%)")
print(f"F1 Score:  {metrics_before['f1']:.2f}% → {metrics_after['f1']:.2f}% "
      f"({metrics_after['f1'] - metrics_before['f1']:+.2f}%)")
print(f"AP:        {metrics_before['AP']:.2f}% → {metrics_after['AP']:.2f}% "
      f"({metrics_after['AP'] - metrics_before['AP']:+.2f}%)")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
print(f"False Positives Removed: {metrics_before['FP'] - metrics_after['FP']}")
print(f"True Positives Removed: {metrics_before['TP'] - metrics_after['TP']}")
print(f"FP Reduction Rate: {(metrics_before['FP'] - metrics_after['FP']) / metrics_before['FP'] * 100:.1f}%" 
      if metrics_before['FP'] > 0 else "N/A")
print(f"TP Retention Rate: {metrics_after['TP'] / metrics_before['TP'] * 100:.1f}%" 
      if metrics_before['TP'] > 0 else "N/A")