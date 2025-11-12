from glob import glob
import os
import pandas as pd
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt


class DriveData(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, root_dir='/dtu/datasets1/02516/DRIVE'):
        'Initialization'
        self.transform = transform
        
        # Always use training folder (has ground truth)
        data_path = os.path.join(root_dir, 'training')
        
        all_image_paths = sorted(glob(data_path + '/images/*.tif'))
        all_mask_paths = sorted(glob(data_path + '/1st_manual/*.gif'))
        
        # Split into train/val/test: 12/4/4
        if split == 'train':
            self.image_paths = all_image_paths[:12]
            self.mask_paths = all_mask_paths[:12]
        elif split == 'val':
            self.image_paths = all_image_paths[12:16]
            self.mask_paths = all_mask_paths[12:16]
        elif split == 'test':
            self.image_paths = all_image_paths[16:20]
            self.mask_paths = all_mask_paths[16:20]
       
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')
        
        # Apply transform to both image and mask
        X = self.transform(image) if self.transform else T.ToTensor()(image)
        Y = self.transform(mask) if self.transform else T.ToTensor()(mask)
        
        # Binarize mask after transform
        Y = (Y > 0.5).long().squeeze(0)
        
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
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        else: 
            image = T.ToTensor()(image)
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().numpy() 
        mask = torch.from_numpy(mask).long()
        
        
        return image, mask







