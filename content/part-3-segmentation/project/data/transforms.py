from torchvision import transforms as T
from torchvision.models import vgg16, VGG16_Weights
import torch
import torchvision.transforms.functional as TF
import random


transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
    ])

# Training transform
transformIMAGENET = T.Compose([
    T.Resize(256),                                # Resize shorter side to 256
    T.RandomCrop(224),                            # Random 224x224 crop
    T.RandomHorizontalFlip(p=0.5),               # 50% chance to flip
    T.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),                                           # RGB jitter
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],      # ImageNet mean
                std=[0.229, 0.224, 0.225])       # ImageNet std
])

# Validation/Test transform
transformtestIMAGENET = VGG16_Weights.IMAGENET1K_V1.transforms()


def flow_transform(flow):
    if random.random() < 0.5:
        flow = TF.hflip(flow)
        flow[0::2] = -flow[0::2]  # Negate x-components
    return flow

# Transform pipeline - flows already in (C, H, W) format
transformFLOW = T.Compose([
    
    T.Lambda(lambda x: torch.from_numpy(x).float()), 
    T.Resize(256),
    
    T.RandomCrop(224),  
    T.Lambda(flow_transform), 
    ])