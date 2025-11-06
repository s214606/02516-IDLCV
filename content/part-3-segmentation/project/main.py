from pipeline import Experiment, TwoStreamFusion
from data.dataloaders import (
    DriveData_trainloader,DriveData_testloader,DriveData_valloader,PH2_testloader,PH2_valloader,PH2_testloader
)
from models.encoder-decoder import Autoencoder
from models.u-net import UNet, UNet2
import torch as t
from config import settings
import torch.optim as optim
from losses.loss import BCELoss, DiceLoss, FocalLoss #yet to submit BCETotalVariation properly

loss_function = t.nn.CrossEntropyLoss()
loss_function = BCELoss()
loss_function = DiceLoss()
loss_function = FocalLoss()

project_name = 'Video Classification'
epochs = 50
dataset = settings.root_dir.split('/')[-1]


drive_cnn = Autoencoder()
drive_cnn_optimizer = t.optim.Adam(drive_cnn.parameters(), lr= 1e-4, weight_decay= 1e-4)

drive_cnn_experiment = Experiment(
    project_name=project_name,
    name = 'Autoencoder-CNN (Retinal Data)',
    config={
        'train_loader': DriveData_trainloader
        'val_loader': DriveData_valloader,
        'test_loader':DriveData_testloader,
        'model': drive_cnn,
        'loss_function': loss_function,
        'optimizer': drive_cnn_optimizer,
        'epochs': epochs,
        'dataset': dataset,
    }
)


drive_cnn_experiment.run()
