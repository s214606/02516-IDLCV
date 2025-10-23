import torch as t

from config import settings
from pipeline import Experiment

from models.single_frame import SingleFrameCNN
from models.early_fusion import EarlyFusion
from models.late_fusion import LateFusion
from models.C3D import C3D
from models.two_stream import DualStreamNetwork

from data.dataloaders import (
    framevideostack_trainloader,
    framevideostack_testloader,
    frameimage_trainloader,
    frameimage_testloader,
)

loss_function = t.nn.CrossEntropyLoss()
project_name = 'Video Classification'
epochs = 50
dataset = settings.root_dir.split('/')[-1]

early_fusion = EarlyFusion(in_size=64)
early_fusion_optimizer = t.optim.Adam(early_fusion.parameters(), lr=1e-4, weight_decay=1e-5)

early_fusion_experiment = Experiment(
    project_name=project_name,
    name='Early Fusion',
    config={
        'train_loader': framevideostack_trainloader,
        'test_loader': framevideostack_testloader,
        'model': early_fusion,
        'loss_function': loss_function,
        'optimizer': early_fusion_optimizer,
        'epochs': epochs,
        'dataset': dataset
    },
    )


late_fusion = LateFusion(
    num_frames=10,
    num_classes=10,
    dropout_rate=0.5,
    fusion='average_pooling'
)

late_fusion_optimizer = t.optim.Adam(late_fusion.parameters(), lr=1e-3, weight_decay=1e-5)

late_fusion_experiment = Experiment(
    project_name=project_name,
    name='Late Fusion',
    config={
        'train_loader': framevideostack_trainloader,
        'test_loader': framevideostack_testloader,
        'model': late_fusion,
        'loss_function': loss_function,
        'optimizer': late_fusion_optimizer,
        'epochs': epochs,
        'dataset': dataset,
    },
    )


single_frame = SingleFrameCNN(
    num_classes=10,
)
single_frame_optimizer = t.optim.Adam(single_frame.parameters(), lr=1e-4, weight_decay=1e-4)

single_frame_experiment = Experiment(
    project_name=project_name,
    name='Single-Frame CNN',
    config={
        'train_loader': frameimage_trainloader,
        'test_loader': frameimage_testloader,
        'model': single_frame,
        'loss_function': loss_function,
        'optimizer': single_frame_optimizer,
        'epochs': epochs,
        'dataset': dataset,
    },
    )

c3d = C3D(
    in_size=128,
    num_classes=10,
    num_frames=10
)

c3d_optimizer = t.optim.Adam(c3d.parameters(), lr=1e-4, weight_decay=1e-5)
c3d_scheduler = t.optim.lr_scheduler.StepLR(
    c3d_optimizer,
    step_size=15,
    gamma=0.1
    )

c3d_experiment = Experiment(
    project_name=project_name,
    name='3D CNN',
    config={
        'train_loader': framevideostack_trainloader,
        'test_loader': framevideostack_testloader,
        'model': c3d,
        'loss_function': loss_function,
        'optimizer': c3d_optimizer,
        'epochs': epochs,
        'dataset': dataset,
        'scheduler': c3d_scheduler,
    },
    )

dual_stream = DualStreamNetwork(
    num_classes=10,
    dropout=0.5,
    temporal=False
    )
dual_stream_optimizer = t.optim.Adam(dual_stream.parameters(), lr=1e-4, weight_decay=1e-5)

dual_stream_experiment = Experiment(
    project_name=project_name,
    name='Dual Stream',
    config={
        'train_loader': frameimage_trainloader,
        'test_loader': frameimage_testloader,
        'model': dual_stream,
        'loss_function': loss_function,
        'optimizer': dual_stream_optimizer,
        'epochs': epochs,
        'dataset': dataset,
    },
    )

from torchvision.models import resnet50, ResNet50_Weights

resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#resnet50_optimizer = t.optim.Adam(resnet50.parameters(), lr=1e-4, weight_decay=1e-5)
resnet_optimizer = t.optim.SGD(resnet.parameters(), lr = 1e-5, momentum=0.9)
resnet_experiment = Experiment(
    project_name=project_name,
    name='ResNet',
    config={
        'train_loader': frameimage_trainloader,
        'test_loader': frameimage_testloader,
        'model': resnet,
        'loss_function': loss_function,
        'optimizer': resnet_optimizer,
        'epochs': epochs,
        'dataset': dataset,
    },
    )

#resnet_experiment.run()
#single_frame_experiment.run()
#early_fusion_experiment.run()
#late_fusion_experiment.run()
c3d_experiment.run()
#dual_stream_experiment.run()