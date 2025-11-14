import torch as t
import torch.nn.functional as F
import wandb
from utils.logger import get_logger
from config import settings
from typing import Dict, Any
from rich.progress import Progress
from metrics.segmentation import Accuracy, IoU, DiceScore
import matplotlib.pyplot as plt
import numpy as np

logger = get_logger(__name__) 
threshold = 0.5
class Experiment:
    """
    Required in config:
    """
    def __init__(
            self, 
            project_name: str,
            name: str,
            config: dict
            ):
        self.project_name = project_name
        self.name = name
        self.config = config
        
        self.progress = Progress()

        self.train_accuracy = Accuracy()
        self.train_dice = DiceScore()
        self.train_iou = IoU()

        self.val_accuracy = Accuracy()
        self.val_dice = DiceScore()
        self.val_iou = IoU()

        self.task = self.progress.add_task(
            f"[red]Running {self.config['epochs']} epochs...",
            total=self.config['epochs']
            )

        self.task_train = self.progress.add_task(
            "[green]Training epoch...",
            total=len(self.config['train_loader'])
            )
        
        self.task_val = self.progress.add_task(
            "[blue]Validating epoch...",
            total=len(self.config['val_loader'])
            )
        
    def _parse_config(self):
        return {k:f'{v=}'.split('=')[0] for k, v in self.config.items()}

    def train(self, epoch):
        self.config['model'].train()
        total_loss = 0.0
        num_batches = 0
        self.train_accuracy.reset()  # Reset accuracy at start of epoch
        
        for X, y in self.config['train_loader']:
            X, y = X.to(settings.device), y.to(settings.device)
            self.config['optimizer'].zero_grad()
            logits = self.config['model'](X)
            loss = self.config['loss_function'](logits, y)
            loss.backward()
            self.config['optimizer'].step()
            
            # Update metrics
            probs = F.sigmoid(logits) 
            logger.info(f"PROBS SHAPE {probs.shape}")
            logger.info("Probs min max: {} / {}".format(probs.min().item(), probs.max().item()))
            self.train_accuracy.update(probs, y, threshold=threshold)
            self.train_dice.update(probs, y, threshold=threshold)
            self.train_iou.update(probs, y, threshold=threshold)
            total_loss += loss.item()
            num_batches += 1

            if num_batches % 2 == 0:
                # Plot predictions to file using matplotlib
                logger.info("Plotting predictions for debugging")
                self._plot_predictions(X, y, probs, num_batches, epoch, phase='train')

            self.progress.update(self.task_train, advance=1)
        
        if self.config.get('scheduler') is not None:
            before_lr = self.config['optimizer'].param_groups[0]['lr']
            self.config['scheduler'].step()
            after_lr = self.config['optimizer'].param_groups[0]['lr']

            if before_lr != after_lr:
                logger.info("Epoch %d: lr %.4f -> %.4f" % (epoch, before_lr, after_lr))


        return {
            'loss/train': total_loss / num_batches,
            'accuracy/train': self.train_accuracy.compute(),
            'dice/train': self.train_dice.compute(),
            'iou/train': self.train_iou.compute()
        }

    def eval(self, epoch):
        self.config['model'].eval()
        total_loss = 0.0
        num_batches = 0
        self.val_accuracy.reset()  # Reset accuracy at start of validation
        
        with t.no_grad():
            for X, y in self.config['val_loader']:
                X, y = X.to(settings.device), y.to(settings.device)
                logits = self.config['model'](X)
                loss = self.config['loss_function'](logits, y)
                
                # Update metrics
                probs = F.sigmoid(logits) 
                self.val_accuracy.update(probs, y, threshold=threshold)
                self.val_dice.update(probs, y, threshold=threshold)
                self.val_iou.update(probs, y, threshold=threshold)
                total_loss += loss.item()
                num_batches += 1
                
                self.progress.update(self.task_val, advance=1)
        
        return {
            'loss/validation': total_loss / num_batches,
            'accuracy/validation': self.val_accuracy.compute(),
            'dice/validation': self.val_dice.compute(),
            'iou/validation': self.val_iou.compute()
        }

    def run(self):
        logger.info("Initializing Weights & Biases run")
        self.experiment = wandb.init(
            entity = 'IDLCV',
            project = self.project_name,
            name=self.name,
            config = self.config
        )

        logger.info("Starting experiment")
        self.progress.start()
        
        logger.info(f"Detected device: {settings.device}")
        self.config['model'].to(settings.device)
        logger.info("Moved model to GPU")

        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_results = self.train(epoch)
                test_results = self.eval(epoch)

                self.experiment.log(train_results | test_results)
                
                self.progress.reset(self.task_train)
                self.progress.reset(self.task_val)
                self.progress.update(self.task, advance=1)

        except Exception as e: # TODO: 
            logger.error(f"Experiment run failed with error: {e}")
        
        finally:
            self.progress.stop()
            self.experiment.finish()

    def _plot_predictions(self, X, y, probs, batch_idx, epoch, phase='train'):
        """Plot predictions vs true masks for debugging"""
        # Convert tensors to numpy for plotting
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().squeeze()  # Assuming y has shape [batch, 1, H, W]
        preds_np = (probs.detach().cpu().numpy() > threshold).astype(np.float32)
        
        # Plot first 4 samples in the batch
        n_samples = min(4, X.shape[0])
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Input image
            img = X_np[i].transpose(1, 2, 0)
            if img.shape[2] == 1:  # Grayscale
                img = img.squeeze()
                axes[i, 0].imshow(img, cmap='gray')
            else:  # RGB
                # Denormalize if needed
                if img.min() < 0 or img.max() > 1:
                    img = (img - img.min()) / (img.max() - img.min())
                axes[i, 0].imshow(img)

            axes[i, 0].set_title(f'Input Image')
            axes[i, 0].axis('off')
            
            # True mask
            # y_np has shape [batch, H, W] with no channel dimension
            axes[i, 1].imshow(y_np[i], cmap='gray')
            axes[i, 1].set_title('True Mask')
            axes[i, 1].axis('off')
            
            # Prediction
            # preds_np has shape [batch, 1, H, W] with channel dimension
            pred_mask = preds_np[i, 0] if preds_np.ndim == 4 else preds_np[i]
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'Epoch {epoch}, {phase} batch {batch_idx}', fontsize=16)
        plt.tight_layout()
        
        # Save to file
        plt.savefig(f'./results/debug_{phase}_epoch{epoch}_batch{batch_idx}.png', dpi=100, bbox_inches='tight')
        plt.close()