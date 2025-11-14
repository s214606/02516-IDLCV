# Experiment runner
A python library to package your deep learning experiments and run the with logging

# Pipeline
1. Create dataloader
2. Initialize experiment
3. Configure experiment
4. Run experiment



```python
import exman as em
import torch
from data.dataloaders import PH2DataLoader

model
optimizer(model.parameters())


em = em(project_name='Segmentation',)


```