from glob import glob
import os
import pandas as pd
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation


class DriveData(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, root_dir='/dtu/datasets1/02516/DRIVE'):
        'Initialization'
        self.split = split
        self.transform = transform
       
        # Use ONLY the training folder, split into train/val/test
        data_path = os.path.join(root_dir, 'training')
        all_image_paths = sorted(glob(data_path + '/images/*.tif'))
        all_mask_paths = sorted(glob(data_path + '/1st_manual/*.gif'))  # ← vessel segmentation masks!
        
        # Split: 12 train / 4 val / 4 test (total 20 images)
        if split == 'train':
            self.image_paths = all_image_paths[:12]
            self.mask_paths = all_mask_paths[:12]
        elif split == 'val':
            self.image_paths = all_image_paths[12:16]
            self.mask_paths = all_mask_paths[12:16]
        elif split == 'test':
            self.image_paths = all_image_paths[16:20]
            self.mask_paths = all_mask_paths[16:20]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # Verify matching lengths
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Mismatch in {split} split: {len(self.image_paths)} images but {len(self.mask_paths)} masks")
       
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        # Apply transform to both image and mask
        X = self.transform(image)
        Y = self.transform(mask)
        
        # Binarize mask after transform
        Y = (Y > 0.5).float()
        
        return X, Y
class PH2(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/dtu/datasets1/02516/PH2_Dataset_images',
    split='', 
    transform=None,
    seed = 1
):      
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        all_folders = sorted([f for f in os.listdir(root_dir) 
                            if f.startswith('IMD') and 
                            os.path.isdir(os.path.join(root_dir, f))])
        
        # Set seed 
        np.random.seed(seed)
        np.random.shuffle(all_folders)
        
        # Calculate split indices for 80/10/10
        n_total = len(all_folders)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        # Split the folders
        if split == 'train':
            self.folders = all_folders[:n_train]
        elif split == 'val':
            self.folders = all_folders[n_train:n_train + n_val]
        elif split == 'test':
            self.folders = all_folders[n_train + n_val:]
        
        self.image_paths = []
        self.mask_paths = []
        
        for folder in self.folders:
            folder_path = os.path.join(root_dir, folder)
            image_path = os.path.join(folder_path, f'{folder}_Dermoscopic_Image', f'{folder}.bmp')
            mask_path = os.path.join(folder_path, f'{folder}_lesion', f'{folder}_lesion.bmp')
            
            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        
        return image, mask




class PH2Clicks(torch.utils.data.Dataset):
    def __init__(self, 
                 base_dataset,
                 positive_clicks=5,
                 negative_clicks=5,
                 centered=True,
                 boundary_width=10,
                 seed=42):
        
        self.base_dataset = base_dataset
        self.n_pos = positive_clicks
        self.n_neg = negative_clicks
        self.centered = centered
        self.boundary_width = boundary_width
        self.seed = seed
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        
        # Convert mask to numpy for processing
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        binary_mask = (mask_np > 0).astype(np.uint8)
        
        # Generate clicks with reproducible seed
        rng = np.random.RandomState(self.seed + idx)
        
        # Positive clicks
        if binary_mask.sum() > 0:
            coords_pos = np.argwhere(binary_mask > 0)
            
            if self.centered:
                # Distance transform for center-biased sampling
                dist_map = distance_transform_edt(binary_mask)
                probs = dist_map[binary_mask > 0]
                probs = probs / probs.sum()
                indices = rng.choice(len(coords_pos), min(self.n_pos, len(coords_pos)), 
                                    replace=False, p=probs)
            else:
                # Uniform sampling
                indices = rng.choice(len(coords_pos), min(self.n_pos, len(coords_pos)), 
                                    replace=False)
            
            pos_clicks = coords_pos[indices]
        else:
            pos_clicks = np.zeros((0, 2), dtype=np.int64)
        
        # Negative clicks
        neg_clicks_list = []
        if self.n_neg > 0:
            # Boundary region (60% of negatives)
            n_boundary = int(self.n_neg * 0.6)
            n_far = self.n_neg - n_boundary
            
            if binary_mask.sum() > 0:
                dilated = binary_dilation(binary_mask, iterations=self.boundary_width)
                boundary_region = dilated & ~binary_mask.astype(bool)
                
                # Near-boundary negatives
                if n_boundary > 0 and boundary_region.sum() > 0:
                    coords_boundary = np.argwhere(boundary_region)
                    indices = rng.choice(len(coords_boundary), 
                                       min(n_boundary, len(coords_boundary)), 
                                       replace=False)
                    neg_clicks_list.append(coords_boundary[indices])
                
                # Far background negatives
                if n_far > 0:
                    bg_mask = ~dilated
                    if bg_mask.sum() > 0:
                        coords_bg = np.argwhere(bg_mask)
                        indices = rng.choice(len(coords_bg), 
                                           min(n_far, len(coords_bg)), 
                                           replace=False)
                        neg_clicks_list.append(coords_bg[indices])
            else:
                # No object, sample from entire image
                coords_bg = np.argwhere(np.ones_like(binary_mask, dtype=bool))
                indices = rng.choice(len(coords_bg), 
                                   min(self.n_neg, len(coords_bg)), 
                                   replace=False)
                neg_clicks_list.append(coords_bg[indices])
        
        neg_clicks = np.vstack(neg_clicks_list) if neg_clicks_list else np.zeros((0, 2), dtype=np.int64)
        
        # Convert to tensors
        pos_clicks = torch.from_numpy(pos_clicks).long()
        neg_clicks = torch.from_numpy(neg_clicks).long()
        
        # Create separate channels for positive and negative clicks
        # Assuming image is (C, H, W) format
        if image.dim() == 3:
            C, H, W = image.shape
        else:
            raise ValueError("Expected image to have shape (C, H, W)")
        
        # Create positive click channel
        pos_channel = torch.zeros((1, H, W), dtype=torch.float)
        if len(pos_clicks) > 0:
            pos_channel[0, pos_clicks[:, 0], pos_clicks[:, 1]] = 1.0
        
        # Create negative click channel
        neg_channel = torch.zeros((1, H, W), dtype=torch.float)
        if len(neg_clicks) > 0:
            neg_channel[0, neg_clicks[:, 0], neg_clicks[:, 1]] = 1.0
        
        # Concatenate: [RGB (3 channels), positive clicks (1 channel), negative clicks (1 channel)]
        input_5ch = torch.cat([image, pos_channel, neg_channel], dim=0)  # Shape: (5, H, W)
        
        return input_5ch, mask
    
    @property
    def split(self):
        return self.base_dataset.split


