from glob import glob
import os
import pandas as pd
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt

import selectivesearch
from xml.etree import ElementTree as ET

class RCNNRegionDataset(torch.utils.data.Dataset):
    def __init__(
            self,
    ):
        pass


class PotHoleData(torch.utils.data.Dataset):
    def __init__(
            self,
            split='train',
            transform=None,
            root_dir='/dtu/datasets1/02516/potholes'
            ):
        
        self.split = split
        self.transform = transform
        self.root_dir = root_dir

        all_image_paths = sorted(glob(os.path.join(root_dir, 'images', '*.png')))
        

def visualize_sample(image, boxes):
    image = Image.open(image).convert("RGB")
    plt.imshow(image)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.axis('off')
    plt.savefig("pothole_sample.png")

if __name__ == '__main__':

    name, boxes = read_content("/dtu/datasets1/02516/potholes/annotations/potholes0.xml")
    print("Name:" , name)
    print("Boxes:", boxes)

    image_path = os.path.join('/dtu/datasets1/02516/potholes/images', name)

    #visualize_sample(image_path, boxes)

    img = Image.open(image_path).convert("RGB")
    img = torch.tensor(np.array(img))
    img_lbl, regions = selectivesearch.selective_search(
        img,
        scale=900,
        sigma=0.9,
        min_size=30
        )
    print("Image label:", img_lbl)
    print("Regions:", regions[:10])

    # Plot some of the proposed regions
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(
10, 10))
    ax.imshow(img)
    for i, region in enumerate(regions):
        if i < 100:
            x, y, w, h = region['rect']
            rect = plt.Rectangle((x, y), w, h,
                                 fill=False, color='blue', linewidth=1)
            ax.add_patch(rect)
    plt.axis('off')
    plt.savefig("pothole_selective_search.png")
    # transform = T.Compose([
    #     T.Resize((256, 256)),
    #     T.ToTensor(),
    # ])

    # dataset = PotHoleData(
    #     split='train',
    #     transform=transform,
    #     root_dir='/dtu/datasets1/02516/potholes'
    #     )
    