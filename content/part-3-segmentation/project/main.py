import torch as t

from pipeline import Experiment
from data.dataloaders import (
    DriveData_trainloader,
    DriveData_testloader,
    DriveData_valloader,
    PH2_trainloader,
    PH2_valloader,
    PH2_testloader
)
from models.encoder_decoder import Autoencoder, EncDec
from models.u_net import UNet

from config import settings
from losses.BCELoss import BCELoss
from losses.loss import FocalLoss, DiceLoss

import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",         # backbone
    encoder_weights="imagenet",      # use ImageNet pretrained weights
    in_channels=3,                   # input channels (RGB)
    classes=1,                       # output channels (binary mask)
)

loss_function = DiceLoss() #t.nn.CrossEntropyLoss()
project_name = 'Segmentation'
epochs = 70
dataset = settings.root_dir.split('/')[-1]


# drive_cnn = model#UNet()
# drive_cnn_optimizer = t.optim.Adam(drive_cnn.parameters(), lr= 1e-3)#, weight_decay= 1e-5)
# drive_cnn_scheduler = t.optim.lr_scheduler.StepLR(
#     drive_cnn_optimizer,
#     step_size=20,
#     gamma=0.5
#     )
# drive_cnn_experiment = Experiment(
#     project_name=project_name,
#     name = 'Autoencoder-CNN (Retinal Data)',
#     config={
#         'train_loader': DriveData_trainloader,
#         'val_loader': DriveData_valloader,
#         'test_loader':DriveData_testloader,
#         'model': drive_cnn,
#         'loss_function': loss_function,
#         'optimizer': drive_cnn_optimizer,
#         'epochs': epochs,
#         'dataset': dataset,
#         'scheduler': drive_cnn_scheduler
#     }
# )

ph2 = model#UNet()
ph2_optimizer = t.optim.Adam(ph2.parameters(), lr= 1e-3)#, weight_decay= 1e-5)
ph2_scheduler = t.optim.lr_scheduler.StepLR(
    ph2_optimizer,
    step_size=20,
    gamma=0.5
    )
ph2_experiment = Experiment(
    project_name=project_name,
    name = 'Autoencoder-CNN (PH2 Data)',
    config={
        'train_loader': PH2_trainloader,
        'val_loader': PH2_valloader,
        'test_loader': PH2_testloader,
        'model': ph2,
        'loss_function': loss_function,
        'optimizer': ph2_optimizer,
        'epochs': epochs,
        'dataset': dataset,
        'scheduler': None
    }
)


ph2_experiment.run()
#drive_cnn_experiment.run()
