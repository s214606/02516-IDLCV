import pandas as pd
import matplotlib.pyplot as plt
import os
from data.dataloaders import framevideostack_trainloader
from utils.logger import get_logger

logger = get_logger(__name__)

splits = ['train', 'val', 'test']
output_dir = 'results'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Check dataloader shape
for video_frames, labels in framevideostack_trainloader:
    print(video_frames.shape, labels.shape)  # [batch, channels, number of frames, height, width]
    break

# Collect data from all splits
all_data = {}
for split in splits:
    #path = f'/zhome/bf/3/167804/02516-IDLCV/content/part-2-video/project/data/ufc10/metadata/{split}.csv'
    path = f'/dtu/datasets1/02516/ucf101_noleakage/metadata/{split}.csv'
    
    df = pd.read_csv(path)
    dist = df['action'].value_counts()
    
    all_data[split] = dist
    print(f"{split}:", dist)

# Get all unique action classes
all_classes = set()
for dist in all_data.values():
    all_classes.update(dist.index)
all_classes = sorted(list(all_classes))

# Prepare data for plotting
split_counts = {split: [all_data[split].get(cls, 0) for cls in all_classes] 
                for split in splits}

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

x = range(len(all_classes))
width = 0.25

# Plot bars for each split
for i, split in enumerate(splits):
    offset = width * (i - 1)
    ax.bar([pos + offset for pos in x], split_counts[split], 
           width, label=split.capitalize())

# Customize the plot
ax.set_xlabel('Action class', fontsize=12)
ax.set_ylabel('Number of samples', fontsize=12)
ax.set_title('Data distribution across train, validation, and test splits (same between leakage & no leakage)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(all_classes, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save the plot
output_path = os.path.join(output_dir, 'data_distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

plt.close()