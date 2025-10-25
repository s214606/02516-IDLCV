from glob import glob
import os
import pandas as pd
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms as T


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/work3/ppar/data/ucf101',
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/work3/ppar/data/ucf101', 
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '/work3/ppar/data/ucf101'

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]

class FrameFlowDataset(torch.utils.data.Dataset):
    def __init__(self, 
        root_dir='/dtu/datasets1/02516/ucf101_noleakage', 
        split='train', 
        transform=None,
        stack_frames=True
    ):
        self.n_sampled_frames = 9
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        flow_dirs = sorted(glob(f'{root_dir}/flows/{split}/*/*'))
        self.flow_dirs = self._filter_valid_flows(flow_dirs)
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')

    def _filter_valid_flows(self, flow_dirs):
       
        valid_dirs = []
        
       
        
        for flow_dir in flow_dirs:
            if os.path.exists(flow_dir):
                # Check for flow files with pattern flow_i_j.npy
                all_exist = all(
                    os.path.exists(os.path.join(flow_dir, f"flow_{i}_{i+1}.npy"))
                    for i in range(1, self.n_sampled_frames + 1)
                )
                if all_exist:
                    valid_dirs.append(flow_dir)
        
        if len(valid_dirs) == 0:
            raise ValueError(f"No valid flow directories found with all {self.n_sampled_frames} flow files")
        
        
        return valid_dirs

    def __len__(self):
        return len(self.flow_dirs)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def load_flows(self, flow_dir):
        """Load flows with naming convention flow_i_j.npy (e.g., flow_1_2.npy, flow_2_3.npy)"""
        flows = []
        
        for i in range(1, self.n_sampled_frames + 1):
            flow_file = os.path.join(flow_dir, f"flow_{i}_{i+1}.npy")
            flow = np.load(flow_file)
            
            
            
            flows.append(flow)
        
    
        flow_volume = np.concatenate(flows, axis=2)

        
        return flows

    def __getitem__(self, idx):
        flow_dir = self.flow_dirs[idx]
        video_name = flow_dir.split('/')[-1]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # Load all flows: list of (2, H, W) arrays
        flows = self.load_flows(flow_dir)
        
        # Stack flows along CHANNEL dimension (axis=0): (2, 224, 224) * 9 -> (18, 224, 224)
        flow_volume = np.concatenate(flows, axis=0)  # Changed from axis=2 to axis=0
        
        
        # Apply transform to the entire volume
        if self.transform:
            flow_volume = self.transform(flow_volume)  # (18, 224, 224) -> (18, 224, 224)
        else:
            flow_volume = torch.from_numpy(flow_volume).float()  # Already (C, H, W) format
        
        # flow_volume is now [2L, H, W] = [18, 224, 224]
        
        if self.stack_frames:
            # Reshape to [C, T, H, W] = [2, 9, 224, 224]
            C = 2
            T = self.n_sampled_frames
            flow_volume = flow_volume.reshape(C, T, flow_volume.shape[1], flow_volume.shape[2])
            
            # Flatten to [C*T, H, W] = [18, 224, 224]
            flow_volume = flow_volume.reshape(C * T, flow_volume.shape[2], flow_volume.shape[3])
        
        return flow_volume, label