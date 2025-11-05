from glob import glob
import os
import pandas as pd
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt


class DriveData(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/dtu/datasets1/02516/DRIVE',
    split='training', 
    transform=None
):      
        split_dir = os.path.join(root_dir, split)

        if split == 'training':
            mask_folder = '1st_manual'
            mask_ext = 'gif' 
        else:
            split_dir = os.path.join(root_dir, 'test')
            mask_folder = 'mask'
            mask_ext = 'gif'
        
        image_pattern = f'{split_dir}/images/*.tif'
        mask_pattern = f'{split_dir}/{mask_folder}/*.gif'
        
        all_image_paths = sorted(glob(image_pattern))
        all_mask_paths = sorted(glob(mask_pattern))
        
        # split test folder in half 10 images for 'val' 10 for 'test' 
        if split == 'val':
            mid_point = len(all_image_paths) // 2
            self.image_paths = all_image_paths[:mid_point]
            self.mask_paths = all_mask_paths[:mid_point]
        elif split == 'test':
            mid_point = len(all_image_paths) // 2
            self.image_paths = all_image_paths[mid_point:]
            self.mask_paths = all_mask_paths[mid_point:]
        else:  # training
            self.image_paths = all_image_paths
            self.mask_paths = all_mask_paths


        self.split = split
        self.transform = transform

       
    def __len__(self):
        return len(self.image_paths)

    def load_image(self, path):
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        
        else: 
            image = T.ToTensor()(image)
        return image

    def load_mask(self, path):
        mask = Image.open(path)
        if self.transform:
            mask = self.transform(mask)
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().numpy() 
        mask = torch.from_numpy(mask).long()
        return mask


    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]        
        
        mask = self.load_mask(mask_path)
        image = self.load_image(image_path)
        return image, mask





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







