import torch

from utils.plotter import visualize_data
from data.dataloaders import (
    PH2_trainloader,
    PH2_valloader,
    PH2_train,
    DriveData_trainloader,
    DriveData_valloader,
    DriveData_train
    )
from models.encoder_decoder import Autoencoder

lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Autoencoder().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

img, mask = next(iter(PH2_trainloader))
print(f'Image batch shape: {img.size()}')
print(f'Mask batch shape: {mask.size()}')
print(mask.unique())

# visualize_data(PH2_train, num_samples=5)
# visualize_data(DriveData_train, num_samples=5)

DATA_PATH = '/dtu/datasets1/02516/phc_data'


import glob
import os
import PIL.Image as Image

class PhC(torch.utils.data.Dataset):
    def __init__(self, train, transform):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(DATA_PATH, 'train' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.jpg'))
        self.label_paths = sorted(glob.glob(data_path + '/labels/*.png'))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y
    
size=128
import torchvision.transforms as transforms
train_transform = transforms.Compose([transforms.Resize((size, size)),
                                 transforms.ToTensor()])
    

from torch.utils.data import DataLoader

dl = DataLoader(PhC(train=True, transform=train_transform) , batch_size=32, shuffle=True)
dl = DriveData_trainloader
img, mask = next(iter(dl))
print(f'Dataset size: {len(dl)}')
print(f'Image shape: {img.size()}')
print(f'Mask shape: {mask.size()}')