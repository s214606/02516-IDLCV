import torch as t

from pipeline import Experiment
from data.dataloaders import (
    DriveData_trainloader,
    DriveData_testloader,
    DriveData_valloader,
    PH2_trainloader,
    PH2_valloader,
    PH2_testloader,
    PH2_clicks_trainloader,
    PH2_clicks_valloader,
    PH2_clicks_testloader
)
from models.encoder_decoder import Autoencoder, EncDec
from models.u_net import UNet256

from config import settings
from losses.BCELoss import BCELoss
from losses.loss import FocalLoss, DiceLoss, BCELossWeighted

import segmentation_models_pytorch as smp

from dotenv import load_dotenv
import os

load_dotenv()  # this reads the .env file

api_key = os.environ.get("API_KEY")
if api_key is None:
    raise ValueError("Missing API_KEY environment variable")

# use api_key
model = smp.Unet(
    encoder_name="resnet34",         # backbone
    encoder_weights="imagenet",      # use ImageNet pretrained weights
    in_channels=5,                   # input channels (RGB)
    classes=1,                       # output channels (binary mask)
)
unet = UNet256()
encdec = EncDec()

loss_function = t.nn.BCEWithLogitsLoss() # DiceLoss()
project_name = 'Segmentation'
epochs = 70
dataset = settings.root_dir.split('/')[-1]


# drive_cnn = encdec#UNet()
# drive_cnn_optimizer = t.optim.Adam(drive_cnn.parameters(), lr= 5e-5, weight_decay= 1e-5)
# drive_cnn_scheduler = t.optim.lr_scheduler.StepLR(
#     drive_cnn_optimizer,
#     step_size=70,
#     gamma=0.5
#     )
    
# drive_cnn_experiment = Experiment(
#     project_name=project_name,
#     name = 'Enc-Dec (Retinal Data), Loss: Focal Loss ',
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

# ph2 = unet
# ph2_optimizer = t.optim.Adam(ph2.parameters(), lr= 5e-4)#, weight_decay= 1e-5)
# ph2_scheduler = t.optim.lr_scheduler.StepLR(
#     ph2_optimizer,
#     step_size=20,
#     gamma=0.5
#     )
# ph2_experiment = Experiment(
#     project_name=project_name,
#     name = 'Encoder-Decoder  (PH2 Data)',
#     config={
#         'train_loader': PH2_trainloader,
#         'val_loader': PH2_valloader,
#         'test_loader': PH2_testloader,
#         'model': ph2,
#         'loss_function': loss_function,
#         'optimizer': ph2_optimizer,
#         'epochs': epochs,
#         'dataset': dataset,
#         'scheduler': ph2_scheduler
#     }
# )


# ph2_experiment.run()
#drive_cnn_experiment.run()


ph2_clicks = model#UNet()
ph2_optimizer_clicks = t.optim.Adam(ph2_clicks.parameters(), lr= 1e-3)#, weight_decay= 1e-5)
ph2_scheduler_clicks = t.optim.lr_scheduler.StepLR(
    ph2_optimizer_clicks,
    step_size=20,
    gamma=0.5
    )
ph2_clicks_experiment = Experiment(
    project_name=project_name,
    name = 'U-Net-CNN (PH2 Data with Weak Labels (3 neg and 3 pos))',
    config={
        'train_loader': PH2_clicks_trainloader,
        'val_loader': PH2_clicks_valloader,
        'test_loader': PH2_clicks_testloader,
        'model': ph2_clicks,
        'loss_function': loss_function,
        'optimizer': ph2_optimizer_clicks,
        'epochs': epochs,
        'dataset': dataset,
        'scheduler': None
    }
)


ph2_clicks_experiment.run()