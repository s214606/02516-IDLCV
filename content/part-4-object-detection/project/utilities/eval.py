from data.dataloaders import test_loader, train_loader, eval_loader
import torch as t
from config import settings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
from collections import defaultdict

image_dir = '/dtu/datasets1/02516/potholes/images'

# Step 1: Collect all bounding boxes grouped by image
print("Collecting all bounding boxes from dataset...")
image_data = defaultdict(lambda: {'bboxes': [], 'labels': []})

with t.no_grad():
    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{len(eval_loader)}")
        
        bboxes = batch['bbox']  # [batch_size, 4]
        labels = batch['label']  # [batch_size]
        image_names = batch['image_name']  # [batch_size]
        
        # Group by image name
        for idx in range(len(image_names)):
            img_name = image_names[idx]
            image_data[img_name]['bboxes'].append(bboxes[idx].cpu())
            image_data[img_name]['labels'].append(labels[idx].cpu().item())

print(f"\nFound {len(image_data)} unique images")
print(f"Total proposals: {sum(len(v['bboxes']) for v in image_data.values())}")

# Step 2: Plot each image with all its bounding boxes
for img_idx, (img_name, data) in enumerate(image_data.items()):

    if True:#img_name == 'potholes0.png':
        print(f"\nPlotting image {img_idx + 1}/{len(image_data)}: {img_name}")
        print(f"  Number of proposals: {len(data['bboxes'])}")
        print(f"  Positive proposals: {sum(1 for l in data['labels'] if l > 0)}")
        print(f"  Negative proposals: {sum(1 for l in data['labels'] if l == 0)}")

        # Load original image
        image_path = os.path.join(image_dir, img_name)
        img = Image.open(image_path).convert("RGB")

        # Create figure
        fig, ax = plt.subplots(1, figsize=(16, 12))
        ax.imshow(img)

        resize_ratio = 2.0  # Scale back to original size

        # Plot all bounding boxes for this image
        num_positive = 0
        num_negative = 0

        for bbox, label in zip(data['bboxes'], data['labels']):
            x1, y1, x2, y2 = bbox * resize_ratio
            width = x2 - x1
            height = y2 - y1

            # Color and style based on label
            if label == 0:
                color = 'red'
                alpha = 0.3
                linewidth = 6
                num_negative += 1
            else:
                color = 'green'
                alpha = 0.8
                linewidth = 10
                num_positive += 1

            rect = Rectangle(
                (x1, y1), width, height,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none',
                alpha=alpha
            )
            ax.add_patch(rect)

            # Only label positive detections to avoid clutter
            if label == 1:
                ax.text(
                    x1, y1 - 5,
                    f"Pothole",
                    color=color,
                    fontsize=9,
                    weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.3')
                )
            else:
                # TODO: Should we add label to the background?
                pass
            
        # Add title with statistics
        ax.set_title(
            f"{img_name}\n"
            f"Positive: {num_positive} | Negative: {num_negative} | Total: {len(data['bboxes'])}",
            fontsize=14,
            weight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        # Save with sanitized filename
        safe_name = img_name.replace('/', '_').replace('\\', '_')
        output_path = f'detections_{safe_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")

        # Optional: only plot first N images to avoid too many files
        if img_idx >= 4:  # Plot only first 5 images
            print(f"\nStopping after {img_idx + 1} images (remove this limit if you want all)")
            break