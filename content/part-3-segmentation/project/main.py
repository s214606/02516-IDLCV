import torch as t

from pipeline import Experiment
from data.dataloaders import (
    DriveData_trainloader,
    DriveData_testloader,
    DriveData_valloader,
    PH2_testloader,
    PH2_valloader,
    PH2_testloader
)
from models.encoder_decoder import Autoencoder, EncDec
from config import settings
from losses.BCELoss import BCELoss



loss_function = BCELoss() #t.nn.CrossEntropyLoss()
project_name = 'Segmentation'
epochs = 70
dataset = settings.root_dir.split('/')[-1]


drive_cnn = EncDec()
drive_cnn_optimizer = t.optim.Adam(drive_cnn.parameters(), lr= 1e-3)#, weight_decay= 1e-5)
drive_cnn_scheduler = t.optim.lr_scheduler.StepLR(
    drive_cnn_optimizer,
    step_size=25,
    gamma=1
    )
drive_cnn_experiment = Experiment(
    project_name=project_name,
    name = 'Autoencoder-CNN (Retinal Data)',
    config={
        'train_loader': DriveData_trainloader,
        'val_loader': DriveData_valloader,
        'test_loader':DriveData_testloader,
        'model': drive_cnn,
        'loss_function': loss_function,
        'optimizer': drive_cnn_optimizer,
        'epochs': epochs,
        'dataset': dataset,
        'scheduler': None
    }
)


drive_cnn_experiment.run()
