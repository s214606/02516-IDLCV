from pipeline import Experiment, TwoStreamFusion
from data.dataloaders import (
    framevideostack_trainloader,
    framevideostack_testloader,
    frameimage_trainloader,
    frameimage_testloader,
    frameflow_testloader,
    frameflow_valloader,
    frameflow_trainloader
)
from models.early_fusion import EarlyFusion
#from models.late_fusion import LateFusion
from models.single_frame import SingleFrameCNN
from models.C3D import C3D
from models.two_stream.vgg8 import TemporalStreamVGG, SpatialStreamVGG
import torch as t
from config import settings
import torch.optim as optim

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


# late_fusion = LateFusion(
#     num_frames=10,
#     num_classes=10,
#     dropout_rate=0.5,
#     fusion='average_pooling'
# )

# late_fusion_optimizer = t.optim.Adam(late_fusion.parameters(), lr=1e-3, weight_decay=1e-5)

# late_fusion_experiment = Experiment(
#     project_name=project_name,
#     name='Late Fusion',
#     config={
#         'train_loader': framevideostack_trainloader,
#         'test_loader': framevideostack_testloader,
#         'model': late_fusion,
#         'loss_function': loss_function,
#         'optimizer': late_fusion_optimizer,
#         'epochs': epochs,
#         'dataset': dataset,
#     },
#     )


single_frame = SingleFrameCNN(
    num_classes=10,
)
single_frame_optimizer = t.optim.Adam(single_frame.parameters(), lr=1e-4, weight_decay=1e-5)

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
    in_size=64,
    num_classes=10,
    num_frames=10
)

c3d_optimizer = t.optim.Adam(c3d.parameters(), lr=1e-4, weight_decay=1e-5)

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
    },
    )


spatial_stream = SpatialStreamVGG(num_classes=10)
spatial_optimizer = t.optim.SGD(spatial_stream.parameters(), lr=5e-3, momentum=0.9)
spatial_scheduler = optim.lr_scheduler.StepLR(spatial_optimizer, step_size=10, gamma=0.1)

spatial_experiment = Experiment(
    project_name=project_name,
    name='Two-Stream Spatial (RGB)',
    config={
        'train_loader': frameimage_trainloader,  # RGB frames only
        'test_loader': frameimage_testloader,
        'model': spatial_stream,
        'loss_function': loss_function,
        'optimizer': spatial_optimizer,
        'epochs': epochs,
        'dataset': dataset,
    },
)

temporal_stream = TemporalStreamVGG(num_classes=10, num_frames=9)
temporal_optimizer = t.optim.SGD(temporal_stream.parameters(), lr=5e-3, momentum=0.9,weight_decay= 1e-4)
temporal_scheduler = optim.lr_scheduler.StepLR(temporal_optimizer, step_size=25, gamma=0.1)

temporal_experiment = Experiment(
    project_name=project_name,
    name='Two-Stream Temporal (Flow)',
    config={
        'train_loader': frameflow_trainloader,  # Optical flow only
        'test_loader': frameflow_valloader,
        'model': temporal_stream,
        'loss_function': loss_function,
        'optimizer': temporal_optimizer,
        'epochs': epochs,
        'dataset': dataset,
        'scheduler':temporal_scheduler,
    },
)



# Load the checkpoints 


# temporal_stream_ = TemporalStreamVGG()
# temporal_stream_.load_state_dict(t.load('checkpoints/temporal_stream.pth'))
# temporal_model = temporal_stream_  # Use the model itself, not the return value

# spatial_stream_ = SpatialStreamVGG()
# spatial_stream_.load_state_dict(t.load('checkpoints/spatial_stream.pth'))
# spatial_model = spatial_stream_  # Use the model itself, not the return value

# two_stream_fusion_vgg_experiment = TwoStreamFusion(
#     project_name=project_name,
#     name='Two-Stream Fused model',
#     config={
#         'spatial_model': spatial_model,
#         'temporal_model': temporal_model,
#         'frame_test_loader': framevideostack_testloader,
#         'flow_test_loader': frameflow_valloader,
#         'loss_function': loss_function
#     }
# )
    



#spatial_experiment.run()
#t.save(spatial_stream.state_dict(), 'checkpoints/spatial_stream_ultimate_test.pth')

#two_stream_fusion_vgg_experiment.run()
temporal_experiment.run()
t.save(temporal_stream.state_dict(), 'checkpoints/temporal_stream_final2.pth')

#single_frame_experiment.run()
#early_fusion_experiment.run()
#late_fusion_experiment.run()
#c3d_experiment.run()