import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
import pandas as pd 
from torch.optim.lr_scheduler import StepLR
import os

# Assuming model, datasets are available
from models.early_fusion import SingleFrameCNN
from datasets import FrameImageDataset
from torch.utils.data import DataLoader

# --- Configuration ---
root_dir = '/zhome/2d/5/168631/Documents/code/DLCV/Project 2/ufc10'
checkpoint_dir = '/results/single-frame/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

classes_list = ['Body-WeightSquats', 'HandstandPushups', 'HandstandWalking', 'JumpingJack', 'JumpRope',
'Lunges', 'PullUps', 'PushUps', 'TrampolineJumping', 'WallPushups']
num_classes = len(classes_list)
n_classes = num_classes 

# --- Data Loading ---
train_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
val_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# --- Model Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SingleFrameCNN(n_classes).to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# --- Modified Training and Validation Functions for per-class accuracy ---

def train_epoch(model, loader, criterion, optimizer, scheduler, device, num_classes):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize counters for per-class accuracy
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Calculate per-class accuracy
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    scheduler.step()
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    # Calculate per-class accuracy
    class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc.append(100 * class_correct[i] / class_total[i])
        else:
            class_acc.append(0.0)
    
    return epoch_loss, epoch_acc, class_acc

def validate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize counters for per-class accuracy
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    # Calculate per-class accuracy
    class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc.append(100 * class_correct[i] / class_total[i])
        else:
            class_acc.append(0.0)
    
    return epoch_loss, epoch_acc, class_acc

# -------------------------------------------------------------
# --- Training Loop and CSV Extraction/Creation ---
# -------------------------------------------------------------

# List to hold metrics for CSV logging
metrics_log = []

num_epochs = 50
best_val_acc = 0.0
print(f"Starting training on device: {device}")

for epoch in range(num_epochs):
    # Train and Validate
    train_loss, train_acc, train_class_acc = train_epoch(model, train_loader, criterion,optimizer, scheduler, device, num_classes)
    val_loss, val_acc, val_class_acc = validate(model, val_loader, criterion, device, num_classes)
    
    # Create epoch log dictionary
    epoch_log = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    
    # Add per-class accuracies to the log
    for i, class_name in enumerate(classes_list):
        epoch_log[f'train_acc_{class_name}'] = train_class_acc[i]
        epoch_log[f'val_acc_{class_name}'] = val_class_acc[i]
    
    metrics_log.append(epoch_log)
    
    # Print results to console
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print('-' * 50)
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, best_model_path)
        print(f'New best model saved with val_acc: {val_acc:.2f}%')

# Save final model
final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'train_acc': train_acc,
    'val_loss': val_loss,
    'val_acc': val_acc,
}, final_model_path)
print(f'Final model saved: {final_model_path}')

# -------------------------------------------------------------
# --- CSV File Creation / Extraction ---
# -------------------------------------------------------------

# Convert the list of dictionaries to a pandas DataFrame
df_metrics = pd.DataFrame(metrics_log)

# Define the output file name
output_csv_file = 'training_metrics.csv'

# Save the DataFrame to a CSV file
df_metrics.to_csv(output_csv_file, index=False)

print(f"Training metrics with per-class accuracy saved to: {output_csv_file}")