import torch as t
from pipeline import Experiment
from models.cnn import FastRCNN
from models.classifier import RCNN_Classifier
from losses.loss import FastRCNNLoss
from config import settings
from data.dataloaders import train_loader, val_loader
project_name = '02516-Object Detection'
epochs = 50


dataset = settings.root_dir.split('/')[-1]
model = RCNN_Classifier(num_classes=2)

#Multi loss BCE + smoothL1
loss_function = FastRCNNLoss(lambda_reg = 1)

#SGD optimizer
rcnn_optimizer = t.optim.SGD(model.parameters(), lr= 1e-3, momentum = 0.9, weight_decay= 1e-5)

#Learning rate scheduler 
rcnn_scheduler = t.optim.lr_scheduler.StepLR(
    rcnn_optimizer,
    step_size = 60,
    gamma = 0.1
)


rcnn_experiment = Experiment(
    project_name= project_name,
    name = 'Fast-RCNN, resnet50',
    config = {
        'train_loader': train_loader, 
        'val_loader': val_loader,     
        #'test_loader': val_dataset,    
        'model': model,
        'loss_function': loss_function, 
        'optimizer': rcnn_optimizer,     
        'epochs': epochs,
        #'dataset': dataset,             
        'scheduler': rcnn_scheduler
    }
)


rcnn_experiment.run()