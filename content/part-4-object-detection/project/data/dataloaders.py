import os
from torch.utils.data import DataLoader
import torch
# from data.datasets import DriveData,PH2, PH2Clicks
from torch.utils.data import DataLoader, random_split
# from data.transforms import train_transform, test_transform
#from config import settings
from data.preprocessing import RCNNDataset

root_dir = "/dtu/blackhole/04/223556/DLCV_p4"

full_dataset = RCNNDataset(
    regions_dir=root_dir, 
    transform=None,       
    balance_classes=True  
)

# Create Train/Val Split 80-20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42) 
)

print(f"Total regions: {len(full_dataset)}")
print(f"Training regions: {len(train_dataset)}")
print(f"Validation regions: {len(val_dataset)}")

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


if __name__ == '__main__':
    print("\n--- Inspecting One Batch ---")

    # Fetch one batch
    batch = next(iter(train_loader))

    images = batch['image']
    labels = batch['label']
    targets = batch['bbox_target']

    print(f"Image Batch Shape:  {images.shape}")   # Expect (128, 3, 227, 227)
    print(f"Labels Batch Shape: {labels.shape}")   # Expect (128)
    print(f"Targets Batch Shape:{targets.shape}")  # Expect (128, 4)

    # Sanity check types
    print(f"Image Type: {images.dtype}")           # Should be torch.float32
    print(f"Label Type: {labels.dtype}")           # Should be torch.int64 (long)

    # # Print dataset information
    # print("=" * 60)
    # print("DATASET INFORMATION")
    # print("=" * 60)

    # print(f"Number of samples: {len(DriveData_train)}")
    # print(f"Split: {DriveData_train.split}")
    # print(f"\nFirst few image paths:")
    # for i, path in enumerate(DriveData_train.image_paths[:3]):
    #     print(f"  {i}: {os.path.basename(path)}")
    # print(f"\nFirst few mask paths:")
    # image, mask = DriveData_train[0]

    # # Print sample information
    # print("\n" + "=" * 60)
    # print("FIRST SAMPLE INFORMATION")
    # print("=" * 60)
    # print(f"Image shape: {image.shape}")
    # print(f"Image dtype: {image.dtype}")
    # print(f"Image min/max: {image.min():.4f} / {image.max():.4f}")
    # print(f"\nMask shape: {mask.shape}")