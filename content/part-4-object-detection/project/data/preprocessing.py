import os
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import selectivesearch

from .utils import read_content
from .transforms import region_transform


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    Boxes are in format [x1, y1, x2, y2]
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def compute_regression_targets(proposal, gt_box):
    """
    Compute bbox regression targets (tx, ty, tw, th).
    proposal and gt_box are in format [x1, y1, x2, y2]
    """
    px = (proposal[0] + proposal[2]) / 2.0
    py = (proposal[1] + proposal[3]) / 2.0
    pw = proposal[2] - proposal[0]
    ph = proposal[3] - proposal[1]
    
    gx = (gt_box[0] + gt_box[2]) / 2.0
    gy = (gt_box[1] + gt_box[3]) / 2.0
    gw = gt_box[2] - gt_box[0]
    gh = gt_box[3] - gt_box[1]
    
    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = np.log(gw / pw)
    th = np.log(gh / ph)
    
    return np.array([tx, ty, tw, th], dtype=np.float32)


def generate_training_data(xml_dir, output_dir, iou_pos_threshold=0.5, 
                          iou_neg_threshold=0.3, max_proposals=2000):
    """
    Generate and save preprocessed training regions with labels.
    
    Args:
        xml_dir: Directory containing XML annotations
        output_dir: Directory to save preprocessed regions
        iou_pos_threshold: IoU threshold for positive examples (default: 0.5)
        iou_neg_threshold: IoU threshold below which examples are negative (default: 0.3)
        max_proposals: Maximum number of proposals to keep per image
    """
    from .utils import read_content  # Import here
    
    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = sorted(glob(os.path.join(xml_dir, '*.xml')))
    
    all_regions = []
    region_id = 0
    
    for fname in xml_files:
        print(f"Processing {os.path.basename(fname)}...")
        
        # Read ground truth
        name, gt_boxes = read_content(fname)  # gt_boxes should be [x1, y1, x2, y2, class_id]
        
        # Load and resize image
        image_path = os.path.join(os.path.dirname(xml_dir), 'images', name)
        img = Image.open(image_path).convert("RGB")
        original_size = img.size
        img = img.resize((img.width // 2, img.height // 2))
        resize_ratio = img.width / original_size[0]
        
        # Adjust ground truth boxes to match resized image
        gt_boxes_resized = gt_boxes.copy()
        gt_boxes_resized[:, :4] *= resize_ratio
        
        # Generate region proposals
        img_np = np.array(img)
        img_lbl, regions = selectivesearch.selective_search(
            img_np,
            scale=900,
            sigma=0.9,
            min_size=10
        )
        
        # Filter and convert proposals to [x1, y1, x2, y2] format
        proposals = []
        for r in regions:
            if r['size'] < 200:
                continue
            x, y, w, h = r['rect']
            proposals.append([x, y, x + w, y + h])
        
        # Limit number of proposals
        if len(proposals) > max_proposals:
            proposals = proposals[:max_proposals]
        
        proposals = np.array(proposals)
        
        # Label each proposal
        for proposal in proposals:
            # Compute IoU with all ground truth boxes
            ious = np.array([compute_iou(proposal, gt_box[:4]) 
                            for gt_box in gt_boxes_resized])
            max_iou = np.max(ious) if len(ious) > 0 else 0
            max_iou_idx = np.argmax(ious) if len(ious) > 0 else -1
            
            # Determine label
            label = 0  # background
            bbox_target = np.zeros(4, dtype=np.float32)
            
            if max_iou >= iou_pos_threshold:
                # Positive example
                label = int(gt_boxes_resized[max_iou_idx, 4])  # class id
                bbox_target = compute_regression_targets(
                    proposal, 
                    gt_boxes_resized[max_iou_idx, :4]
                )
            elif max_iou < iou_neg_threshold:
                # Negative example (background)
                label = 0
            else:
                # Skip ambiguous examples (between thresholds)
                continue
            
            # Crop and resize region to 227x227
            x1, y1, x2, y2 = proposal.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            
            region_img = img.crop((x1, y1, x2, y2))
            region_img = region_img.resize((227, 227))
            
            # Convert to tensor (C, H, W) and normalize to [0, 1]
            region_tensor = torch.from_numpy(np.array(region_img)).permute(2, 0, 1).float() / 255.0
            
            # Save as single .pt file with all data
            region_data = {
                'image': region_tensor,
                'label': torch.tensor(label, dtype=torch.long),
                'bbox_target': torch.tensor(bbox_target, dtype=torch.float32),
                'bbox': torch.tensor(proposal, dtype=torch.float32),
                'image_name': name
            }
            
            save_path = os.path.join(output_dir, f'region_{region_id:06d}.pt')
            torch.save(region_data, save_path)
            
            all_regions.append({'region_id': region_id, 'label': label})
            region_id += 1
    
    # Save index of all regions (just IDs and labels for quick access)
    index_path = os.path.join(output_dir, 'index.pt')
    torch.save(all_regions, index_path)
    
    print(f"Generated {len(all_regions)} training regions")
    print(f"Positive examples: {sum(1 for r in all_regions if r['label'] > 0)}")
    print(f"Negative examples: {sum(1 for r in all_regions if r['label'] == 0)}")


class RCNNDataset(Dataset):
    """
    Dataset for R-CNN training.
    Loads preprocessed region proposals with labels.
    """
    def __init__(self, regions_dir, transform=None, balance_classes=True):
        """
        Args:
            regions_dir: Directory containing preprocessed regions
            transform: Optional transform to apply to images (applied to already-normalized tensors)
            balance_classes: Whether to balance positive/negative examples
        """
        self.regions_dir = regions_dir
        self.transform = transform
        
        # Load index
        index_path = os.path.join(regions_dir, 'index.pt')
        self.regions = torch.load(index_path)
        
        # Optional: balance classes
        if balance_classes:
            self.regions = self._balance_classes(self.regions)
        
        print(f"Loaded {len(self.regions)} regions")
    
    def _balance_classes(self, regions):
        """Balance positive and negative examples."""
        positive = [r for r in regions if r['label'] > 0]
        negative = [r for r in regions if r['label'] == 0]
        
        # Keep all positives, sample negatives to match
        n_positive = len(positive)
        n_negative = len(negative)
        
        if n_negative > n_positive:
            # Randomly sample negatives
            neg_indices = np.random.choice(n_negative, n_positive, replace=False)
            negative = [negative[i] for i in neg_indices]
        
        balanced = positive + negative
        np.random.shuffle(balanced)
        
        print(f"Balanced dataset: {len(positive)} positive, {len(negative)} negative")
        return balanced
    
    def __len__(self):
        return len(self.regions)
    
    def __getitem__(self, idx):
        region_info = self.regions[idx]
        region_id = region_info['region_id']
        
        # Load preprocessed region (already a tensor at 227x227)
        region_path = os.path.join(self.regions_dir, f'region_{region_id:06d}.pt')
        region_data = torch.load(region_path)
        
        image = region_data['image']  # Already (3, 227, 227) tensor in [0, 1]
        
        # Apply additional transforms if needed (e.g., normalization)
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': region_data['label'],
            'bbox_target': region_data['bbox_target']
        }


class RCNNTestDataset(Dataset):
    """
    Dataset for R-CNN testing.
    Returns all proposals for each image.
    """
    def __init__(self, xml_dir, transform=None):
        from .utils import read_content
        self.xml_dir = xml_dir
        self.transform = transform
        self.xml_files = sorted(glob(os.path.join(xml_dir, '*.xml')))
        self.read_content = read_content
    
    def __len__(self):
        return len(self.xml_files)
    
    def __getitem__(self, idx):
        fname = self.xml_files[idx]
        
        # Read image
        name, _ = self.read_content(fname)
        image_path = os.path.join(os.path.dirname(self.xml_dir), 'images', name)
        img = Image.open(image_path).convert("RGB")
        img = img.resize((img.width // 2, img.height // 2))
        
        # Generate proposals
        img_np = np.array(img)
        img_lbl, regions = selectivesearch.selective_search(
            img_np,
            scale=900,
            sigma=0.9,
            min_size=10
        )
        
        # Filter proposals
        proposals = []
        for r in regions:
            if r['size'] < 200:
                continue
            x, y, w, h = r['rect']
            proposals.append([x, y, x + w, y + h])
        
        proposals = np.array(proposals)
        
        # Crop and resize each proposal
        regions_list = []
        for proposal in proposals:
            x1, y1, x2, y2 = proposal.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            
            region_img = img.crop((x1, y1, x2, y2))
            region_img = region_img.resize((227, 227))
            
            if self.transform:
                region_img = self.transform(region_img)
            else:
                region_img = torch.from_numpy(np.array(region_img)).permute(2, 0, 1).float() / 255.0
            
            regions_list.append(region_img)
        
        regions_tensor = torch.stack(regions_list)
        proposals_tensor = torch.from_numpy(proposals).float()
        
        return {
            'regions': regions_tensor,
            'proposals': proposals_tensor,
            'image_name': name,
            'original_image': img
        }


if __name__ == '__main__':
    # Generate training data
    generate_training_data(
        xml_dir="/dtu/datasets1/02516/potholes/annotations",
        output_dir="./data/rcnn_regions",
        iou_pos_threshold=0.5,
        iou_neg_threshold=0.3
    )