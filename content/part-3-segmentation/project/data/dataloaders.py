from torch.utils.data import DataLoader
from data.datasets import FrameVideoDataset, FrameImageDataset, FrameFlowDataset
from data.transforms import transform, transformIMAGENET, transformtestIMAGENET, transformFLOW
from config import settings

framevideostack_trainset = FrameVideoDataset(
    root_dir=settings.root_dir,
    split='train',
    transform=transformIMAGENET,
    stack_frames = True
    )

framevideostack_testset = FrameVideoDataset(
    root_dir=settings.root_dir,
    split='val',
    transform=transformtestIMAGENET,
    stack_frames = True
    )

framevideostack_trainloader = DataLoader(
    framevideostack_trainset,
    batch_size=32,
    shuffle=True,
    num_workers=4
    )

framevideostack_testloader = DataLoader(
    framevideostack_testset,
    batch_size=32,
    shuffle=False,
    num_workers=4
    )

frameimage_trainset = FrameImageDataset(
    root_dir=settings.root_dir,
    split='train',
    transform=transformIMAGENET
    )

frameimage_testset = FrameImageDataset(
    root_dir=settings.root_dir,
    split='val',
    transform=transformtestIMAGENET
    )

frameimage_trainloader = DataLoader(
    frameimage_trainset, 
    batch_size=64,
    shuffle=True,
    num_workers=4
    )

frameimage_testloader = DataLoader(
    frameimage_testset,
    batch_size=64,
    shuffle=False,
    num_workers=4)


frameflow_trainset = FrameFlowDataset(
    root_dir=settings.root_dir,
    split='train',
    transform=transformFLOW,
    stack_frames=True  # CHANGED: Added this
    )

frameflow_valset = FrameFlowDataset(
    root_dir=settings.root_dir,
    split='val',
    transform=None,
    stack_frames=True  # CHANGED: Added this
    )

frameflow_testset = FrameFlowDataset(
    root_dir=settings.root_dir,
    split='test',  # CHANGED: Changed from 'test' to 'val' - use val as test
    transform=None,
    stack_frames=True  # CHANGED: Added this
    )


frameflow_trainloader = DataLoader(
    frameflow_trainset, 
    batch_size=64,
    shuffle=True,
    num_workers=4
    )

frameflow_valloader = DataLoader(
    frameflow_valset,
    batch_size=64,
    shuffle=False,
    num_workers=0)

frameflow_testloader = DataLoader(
    frameflow_testset,
    batch_size=64,
    shuffle=False,
    num_workers=0)