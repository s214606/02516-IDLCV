import torch as t
from torch.utils.data import DataLoader
from pipeline import Experiment
from models.cnn import FastRCNN
from losses.loss import FastRCNNLoss

# --- 1. Dummy Data Generator ---
def get_dummy_batch(batch_size=2, num_proposals=10, img_size=512, num_classes=20):
    """
    Generates random data simulating the output of a Dataset/DataLoader.
    """
    # 1. Images: (Batch_Size, 3 channels, H, W)
    images = t.randn(batch_size, 3, img_size, img_size)
    
    # 2. Proposals: (Batch_Size, Num_Proposals, 4)
    # Generate random boxes [x1, y1, x2, y2] within image bounds
    # Note: We sort to ensure x1 < x2 and y1 < y2 roughly
    proposals = t.rand(batch_size, num_proposals, 4) * img_size
    proposals[:, :, 2:] += proposals[:, :, :2] # Make sure x2 > x1, y2 > y1
    
    # 3. Labels: (Batch_Size * Num_Proposals)
    # Random integers from 0 (background) to num_classes
    # Flattened because CrossEntropy expects 1D target for the batch
    labels = t.randint(0, num_classes + 1, (batch_size * num_proposals,))
    
    # 4. Target Deltas: (Batch_Size * Num_Proposals, 4)
    # Dummy regression targets (dx, dy, dw, dh)
    target_deltas = t.randn(batch_size * num_proposals, 4)
    
    return images, proposals, labels, target_deltas

# --- 2. Create Dummy Loaders ---
# A list of batches acts exactly like a DataLoader in a for-loop
dummy_batch = get_dummy_batch()
# Create a list containing 5 batches to simulate an epoch
dummy_train_loader = [get_dummy_batch() for _ in range(5)]
dummy_val_loader = [get_dummy_batch() for _ in range(2)]


# --- 3. Setup Experiment ---
project_name = '02516-Object Detection-TEST'
epochs = 2 # Short run for testing

# Initialize Model (Ensure num_classes matches dummy data)
model = FastRCNN(num_classes=20) 

# Initialize Loss
loss_function = FastRCNNLoss(lambda_reg=1.0)

# Optimizer
rcnn_optimizer = t.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)

# Scheduler
rcnn_scheduler = t.optim.lr_scheduler.StepLR(
    rcnn_optimizer,
    step_size=1, # fast step for testing
    gamma=0.1
)

# --- 4. Run Experiment ---
print("🚀 Starting Dry Run with Dummy Data...")

rcnn_experiment = Experiment(
    project_name=project_name,
    name='Fast-RCNN_Test_Run',
    config={
        'train_loader': dummy_train_loader, 
        'val_loader': dummy_val_loader,     
        # 'test_loader': None, # Optional: leave out if not testing
        'model': model,
        'loss_function': loss_function, # PASS THE OBJECT, NOT THE STRING
        'optimizer': rcnn_optimizer,        
        'epochs': epochs,
        'dataset': None, # Not needed for dummy run             
        'scheduler': rcnn_scheduler
    }
)

rcnn_experiment.run()