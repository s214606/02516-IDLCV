import os
from torch.utils.data import DataLoader
import torch
from data.datasets import DriveData,PH2, PH2Clicks
from data.transforms import transform
#from config import settings


DriveData_train = DriveData(split='training', transform = transform)

DriveData_trainloader = DataLoader(
    DriveData_train,
    batch_size = 32,
    shuffle= True,
    num_workers=4
)

DriveData_val = DriveData(split='val',transform = transform)

DriveData_valloader = DataLoader(
    DriveData_val,
    batch_size = 32,
    shuffle= True,
    num_workers=4
)

DriveData_test = DriveData(split='test',transform = transform)

DriveData_testloader = DataLoader(
    DriveData_test,
    batch_size = 32,
    shuffle= True,
    num_workers=4
)



PH2_train = PH2(split='train',transform = transform)

PH2_trainloader = DataLoader(
    PH2_train,
    batch_size = 32,
    shuffle= True,
    num_workers=4
)

PH2_val = PH2(split='val',transform = transform)

PH2_valloader = DataLoader(
    PH2_val,
    batch_size = 32,
    shuffle= True,
    num_workers=4
)

PH2_test = PH2(split='test',transform = transform)

PH2_testloader = DataLoader(
    PH2_test,
    batch_size = 32,
    shuffle= True,
    num_workers=4
)




# Wrap with clicks
PH2_clicks = PH2Clicks(
    base_dataset=PH2_train,
    positive_clicks=3,
    negative_clicks=3,
    centered=False,
    boundary_width=5,
    seed=None
)



if __name__ == '__main__':
    # Print dataset information
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Number of samples: {len(DriveData_train)}")
    print(f"Split: {DriveData_train.split}")
    print(f"\nFirst few image paths:")
    for i, path in enumerate(DriveData_train.image_paths[:3]):
        print(f"  {i}: {os.path.basename(path)}")
    print(f"\nFirst few mask paths:")
    image, mask = DriveData_train[0]

    # Print sample information
    print("\n" + "=" * 60)
    print("FIRST SAMPLE INFORMATION")
    print("=" * 60)
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image min/max: {image.min():.4f} / {image.max():.4f}")
    print(f"\nMask shape: {mask.shape}")


    # Print dataset information
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Number of samples: {len(DriveData_val)}")
    print(f"Split: {DriveData_val.split}")
    print(f"\nFirst few image paths:")
    for i, path in enumerate(DriveData_val.image_paths[:3]):
        print(f"  {i}: {os.path.basename(path)}")
    print(f"\nFirst few mask paths:")



    # Print dataset information
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Number of samples: {len(DriveData_test)}")
    print(f"Split: {DriveData_test.split}")
    print(f"\nFirst few image paths:")
    for i, path in enumerate(DriveData_test.image_paths[:3]):
        print(f"  {i}: {os.path.basename(path)}")
    print(f"\nFirst few mask paths:")

    # Print dataset information
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Number of samples: {len(PH2_train)}")
    print(f"Split: {PH2_train.split}")
    print(f"\nFirst few image paths:")
    for i, path in enumerate(PH2_train.image_paths[:3]):
        print(f"  {i}: {os.path.basename(path)}")
    print(f"\nFirst few mask paths:")



    # Print dataset information
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Number of samples: {len(PH2_val)}")
    print(f"Split: {PH2_val.split}")
    print(f"\nFirst few image paths:")
    for i, path in enumerate(PH2_val.image_paths[:3]):
        print(f"  {i}: {os.path.basename(path)}")
    print(f"\nFirst few mask paths:")



    # Print dataset information
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Number of samples: {len(PH2_test)}")
    print(f"Split: {PH2_test.split}")
    print(f"\nFirst few image paths:")
    for i, path in enumerate(PH2_test.image_paths[:3]):
        print(f"  {i}: {os.path.basename(path)}")
    print(f"\nFirst few mask paths:")




    input_5ch, mask = PH2_clicks[3]

    import matplotlib.pyplot as plt

    # Extract channels
    image = input_5ch[:3]  # RGB channels
    pos_channel = input_5ch[3]  # Positive clicks channel
    neg_channel = input_5ch[4]  # Negative clicks channel

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Image')

    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

    plt.subplot(133)
    plt.imshow(image.permute(1, 2, 0))
    pos_coords = torch.where(pos_channel == 1)
    neg_coords = torch.where(neg_channel == 1)
    plt.scatter(pos_coords[1], pos_coords[0], c='green', s=100, marker='o', label='Positive')
    plt.scatter(neg_coords[1], neg_coords[0], c='red', s=100, marker='x', label='Negative')
    plt.legend()
    plt.title('Clicks')

    plt.savefig('clicks_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()