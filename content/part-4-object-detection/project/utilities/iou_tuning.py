import os
import numpy as np 
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt

import selectivesearch
from data.utils import read_content


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def visualize_sample(image, boxes):
    image = Image.open(image).convert("RGB")
    plt.imshow(image)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.axis('off')
    plt.savefig("pothole_sample.png")


if __name__ == '__main__':

    name, boxes = read_content("/dtu/datasets1/02516/potholes/annotations/potholes127.xml")
    print("Name:" , name)
    print("Boxes:", boxes)

    image_path = os.path.join('/dtu/datasets1/02516/potholes/images', name)

    #visualize_sample(image_path, boxes)

    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    img = img.resize((img.width // 2, img.height // 2))
    resize_ratio = img.width / original_size[0]

    # Adjust ground truth boxes to match resized image
    gt_boxes_resized = np.array(boxes, dtype=float)
    gt_boxes_resized[:, :4] *= resize_ratio

    # Convert to numpy array for selective search
    img_np = np.array(img)

    img_lbl, regions = selectivesearch.selective_search(
        img_np,
        scale=100,
        sigma=0.8,
        min_size=50
    )

    # Filter proposals
    proposals = []
    for r in regions:
        if r['size'] < 50:
            continue
        x, y, w, h = r['rect']
        proposals.append([x, y, x + w, y + h])

    print(f"Total regions after filtering: {len(proposals)}")

    # Compute IoU for each proposal with ground truth boxes
    iou_pos_threshold = 0.35
    iou_neg_threshold = 0.1
    
    positive_proposals = []
    negative_proposals = []
    ambiguous_proposals = []
    
    for proposal in proposals:
        # Compute IoU with all ground truth boxes
        ious = np.array([compute_iou(proposal, gt_box[:4]) 
                        for gt_box in gt_boxes_resized])
        max_iou = np.max(ious) if len(ious) > 0 else 0
        max_iou_idx = np.argmax(ious) if len(ious) > 0 else -1
        
        # Classify proposal based on IoU
        if max_iou >= iou_pos_threshold:
            positive_proposals.append((proposal, max_iou, int(gt_boxes_resized[max_iou_idx, 4])))
        elif max_iou < iou_neg_threshold:
            negative_proposals.append((proposal, max_iou))
        else:
            ambiguous_proposals.append((proposal, max_iou))
    
    print(f"\nProposal Statistics:")
    print(f"Positive proposals (IoU >= {iou_pos_threshold}): {len(positive_proposals)}")
    print(f"Negative proposals (IoU < {iou_neg_threshold}): {len(negative_proposals)}")
    print(f"Ambiguous proposals: {len(ambiguous_proposals)}")

    # Visualize with color coding
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.imshow(img_np)
    
    # Draw ground truth boxes in red
    #for gt_box in gt_boxes_resized:
    #    x1, y1, x2, y2 = gt_box[:4]
    #    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
    #                         fill=False, color='red', linewidth=3, label='Ground Truth')
    #    ax.add_patch(rect)
    
    # Draw positive proposals in green (limit to first 50 for visibility)
    for proposal, iou in ambiguous_proposals[:50]:
        x1, y1, x2, y2 = proposal
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, color='yellow', linewidth=5, alpha=0.4)
        ax.add_patch(rect)

    for proposal, iou, class_id in positive_proposals[:50]:
        x1, y1, x2, y2 = proposal
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, color='green', linewidth=5, alpha=0.7)
        ax.add_patch(rect)
    
    # Draw negative proposals in blue (limit to first 50)
    for proposal, iou in negative_proposals[:50]:
        x1, y1, x2, y2 = proposal
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, color='blue', linewidth=1, alpha=0.5)
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.title(f'Red: GT | Green: Positive (IoU≥{iou_pos_threshold}) | Blue: Negative (IoU<{iou_neg_threshold})')
    plt.savefig("results/iou_tunings/pothole_selective_search_iou.png", dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to pothole_selective_search_iou.png")