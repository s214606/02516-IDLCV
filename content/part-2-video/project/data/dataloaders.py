from torch.utils.data import DataLoader
from data.datasets import FrameVideoDataset
from data.transforms import transform
from config import settings

framevideostack_trainset = FrameVideoDataset(
    root_dir=settings.root_dir,
    split='train',
    transform=transform,
    stack_frames = True
    )

framevideostack_testset = FrameVideoDataset(
    root_dir=settings.root_dir,
    split='test',
    transform=transform,
    stack_frames = True
    )

framevideostack_trainloader = DataLoader(
    framevideostack_trainset,
    batch_size=8,
    shuffle=True
    )

framevideostack_testloader = DataLoader(
    framevideostack_testset,
    batch_size=8,
    shuffle=False
    )