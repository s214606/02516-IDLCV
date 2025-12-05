from glob import glob
import re
import os

import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from data.utils import read_content

def visualize_region_proposals(xml_dir, regions_dir, image_dir, output_dir='visualizations',
                               image_slice=slice(None), max_regions_per_image=50):
    """
    Visualize region proposals with binary classification (background vs pothole).
    
    Args:
        regions_dir: Directory containing preprocessed region .pt files
        image_dir: Directory containing original images
        output_dir: Directory to save visualization images
        image_slice: Slice object to select images (e.g., slice(0, 3) for first 3)
        max_regions_per_image: Maximum number of regions to draw per image
    """
    
    xml_files = sorted(
        glob(os.path.join(xml_dir, '*.xml')),
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load index
    index_path = Path(regions_dir) / 'index.pt'
    index = torch.load(index_path)
    
    # Group regions by image name
    image_regions = defaultdict(list)
    
    print("Loading and grouping regions by image...")
    for idx in index:
        region_id = idx['region_id']
        region_path = Path(regions_dir) / f"region_{region_id:06d}.pt"
        region_data = torch.load(region_path)
        
        image_name = region_data['image_name']
        # Convert any positive label to 1 (pothole)
        binary_label = 1 if region_data['label'].item() > 0 else 0
        
        image_regions[image_name].append({
            'bbox': region_data['bbox'].numpy(),
            'label': binary_label,
            'region_id': region_id
        })
    
    # Get sorted list of unique image names
    image_names = sorted(image_regions.keys())
    
    # Apply slice
    selected_images = image_names[image_slice]
    
    print(f"\nFound {len(image_names)} unique images")
    print(f"Visualizing {len(selected_images)} images")
    
    # Create visualization for each selected image
    for img_idx, image_name in enumerate(selected_images):
        print(f"\nProcessing {image_name} ({img_idx + 1}/{len(selected_images)})...")
        
        # Load metadata - fix: use .xml extension
        xml_path = Path(xml_dir) / f"{image_name.split('.')[0]}.xml"
        name, gt_boxes = read_content(xml_path)

        # Load original image
        image_path = Path(image_dir) / image_name
        img = Image.open(image_path).convert("RGB")
        
        # Resize to match preprocessing (half size)
        resize_ratio = 0.5
        img_resized = img.resize((img.width // 2, img.height // 2))
        gt_boxes_resized = gt_boxes.copy()
        gt_boxes_resized = np.array(gt_boxes_resized, dtype=float)
        gt_boxes_resized[:, :4] *= resize_ratio
        
        # Create copies for drawing
        img_gt = img_resized.copy()
        draw_gt = ImageDraw.Draw(img_gt)
        
        img_proposals = img_resized.copy()
        draw_proposals = ImageDraw.Draw(img_proposals)
        
        # Draw ground truth boxes on left image
        for gt_box in gt_boxes_resized:
            x1, y1, x2, y2 = gt_box[:4]
            draw_gt.rectangle([x1, y1, x2, y2], outline='blue', width=3)
        
        # Get regions for this image
        regions = image_regions[image_name]
        
        # Limit number of regions to draw
        if len(regions) > max_regions_per_image:
            # Sample diverse regions (mix of background and pothole)
            pothole_regions = [r for r in regions if r['label'] == 1]
            background_regions = [r for r in regions if r['label'] == 0]
            
            n_pothole = min(len(pothole_regions), max_regions_per_image // 2)
            n_background = min(len(background_regions), max_regions_per_image - n_pothole)
            
            sampled_regions = (pothole_regions[:n_pothole] + 
                             background_regions[:n_background])
        else:
            sampled_regions = regions
        
        # Count labels
        n_background = sum(1 for r in regions if r['label'] == 0)
        n_pothole = sum(1 for r in regions if r['label'] == 1)
        
        # Draw bounding boxes - background first, then potholes on top
        # Separate regions by class
        background_regions = [r for r in sampled_regions if r['label'] == 0]
        pothole_regions = [r for r in sampled_regions if r['label'] == 1]
        
        # Draw background regions first (will be under potholes)
        for region in background_regions:
            bbox = region['bbox']
            x1, y1, x2, y2 = bbox
            draw_proposals.rectangle([x1, y1, x2, y2], outline='red', width=1)
        
        # Draw pothole regions on top
        for region in pothole_regions:
            bbox = region['bbox']
            x1, y1, x2, y2 = bbox
            draw_proposals.rectangle([x1, y1, x2, y2], outline='green', width=2)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Show image with ground truth boxes
        axes[0].imshow(img_gt)
        axes[0].set_title(f'Ground Truth Boxes\n{image_name} (n={len(gt_boxes)})')
        axes[0].axis('off')
        
        # Show image with proposals
        axes[1].imshow(img_proposals)
        title = f'Region Proposals (showing {len(sampled_regions)}/{len(regions)})\n'
        title += f'Pothole: {n_pothole}, Background: {n_background}'
        axes[1].set_title(title)
        axes[1].axis('off')
        
        # Add legends
        from matplotlib.patches import Patch
        legend_elements_gt = [
            Patch(facecolor='blue', label=f'Ground Truth (n={len(gt_boxes)})')
        ]
        axes[0].legend(handles=legend_elements_gt, loc='upper right')
        
        legend_elements_proposals = [
            Patch(facecolor='green', label=f'Pothole (n={n_pothole})'),
            Patch(facecolor='red', label=f'Background (n={n_background})')
        ]
        axes[1].legend(handles=legend_elements_proposals, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure with unique name
        output_filename = output_path / f'proposals_{img_idx:03d}_{Path(image_name).stem}.png'
        plt.savefig(output_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {output_filename}")
        print(f"  Total regions: {len(regions)}")
        print(f"  Pothole: {n_pothole}, Background: {n_background}")


# Example usage
if __name__ == '__main__':
    xml_dir = '/dtu/datasets1/02516/potholes/annotations'
    regions_dir = '/dtu/blackhole/1c/167804/proposals'
    image_dir = '/dtu/datasets1/02516/potholes/images'
    output_dir = 'results/proposal_visualizations'
    
    # Visualize first 3 images
    visualize_region_proposals(
        xml_dir=xml_dir,
        regions_dir=regions_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        image_slice=slice(0, 50),  # First 50 images
        max_regions_per_image=200
    )
    
    # Other examples:
    # visualize_region_proposals(..., image_slice=slice(5, 8))  # Images 5-7
    # visualize_region_proposals(..., image_slice=slice(None, None, 2))  # Every other image
    # visualize_region_proposals(..., image_slice=slice(-3, None))  # Last 3 images