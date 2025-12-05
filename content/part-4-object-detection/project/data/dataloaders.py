import os
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, random_split
from data.preprocessing import RCNNDataset
from data.transforms import region_transform

root_dir = "/dtu/blackhole/1c/167804/proposals"

full_dataset = RCNNDataset(
    regions_dir=root_dir, 
    transform=region_transform,       
    balance_classes=True  
)

eval_dataset = RCNNDataset(
    regions_dir=root_dir, 
    transform=region_transform,       
    balance_classes=True
)

# Create Train/Val Split 80-20
train_size = int(0.8 * len(full_dataset))
val_size = int(0.10 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42) 
)

print(f"Total regions: {len(full_dataset)}")
print(f"Training regions: {len(train_dataset)}")
print(f"Validation regions: {len(val_dataset)}")
print(f"Test regions: {len(test_dataset)}")

# 4. Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,         
    num_workers=4,
    pin_memory=True     
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,         
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,         
    num_workers=4,
    pin_memory=True
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=128,
    shuffle=False,         
    num_workers=4,
    pin_memory=True
)


if __name__ == '__main__':
    print("\n--- Inspecting One Batch ---")

    # Fetch one batch
    batch = next(iter(train_loader))
    #images, proposals, labels, target_deltas = batch
    images = batch['image']
    labels = batch['label']
    targets = batch['bbox_target']
    proposals = batch['bbox']

    print(f"Image Batch Shape:  {images.shape}")   # Expect (128, 3, 227, 227)
    print(f"Labels Batch Shape: {labels.shape}")   # Expect (128)
    print(f"Targets Batch Shape:{targets.shape}")  # Expect (128, 4)
    print(f"Proposals batch shape:{proposals.shape}")
    # Sanity check types
    print(f"Image Type: {images.dtype}")           # Should be torch.float32
    print(f"Label Type: {labels.dtype}")           # Should be torch.int64 (long)


