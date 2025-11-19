import torch as t
import torch.nn.functional as F
import wandb
from utils.logger import get_logger
from config import settings
from typing import Dict, Any
from rich.progress import Progress
from metrics.segmentation import Accuracy, IoU, DiceScore, Sensitivity, Specificity
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
        self.train_sensitivity = Sensitivity()
        self.train_specificity = Specificity()

        self.val_accuracy = Accuracy()
        self.val_dice = DiceScore()
        self.val_iou = IoU()
        self.val_sensitivity = Sensitivity()
        self.val_specificity = Specificity()

        self.test_accuracy = Accuracy()
        self.test_dice = DiceScore()
        self.test_iou = IoU()
        self.test_sensitivity = Sensitivity()
        self.test_specificity = Specificity()
      

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
        
        self.task_test = self.progress.add_task(
            "[yellow]Testing on test set...",
            total=len(self.config.get('test_loader', [])) if self.config.get('test_loader') else 0,
            visible=False  # Hide until test phase
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
            self.train_accuracy.update(probs, y, threshold=threshold)
            self.train_dice.update(probs, y, threshold=threshold)
            self.train_iou.update(probs, y, threshold=threshold)
            self.train_sensitivity.update(probs, y, threshold=threshold)
            self.train_specificity.update(probs, y, threshold=threshold)
            total_loss += loss.item()
            num_batches += 1

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
            'iou/train': self.train_iou.compute(),
            'sensitivity/train':self.train_sensitivity.compute(),
            'specificity/train':self.train_specificity.compute(),
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
                self.val_sensitivity.update(probs, y, threshold=threshold)
                self.val_specificity.update(probs, y, threshold=threshold)
                total_loss += loss.item()
                num_batches += 1
                
                self.progress.update(self.task_val, advance=1)
        
        return {
            'loss/validation': total_loss / num_batches,
            'accuracy/validation': self.val_accuracy.compute(),
            'dice/validation': self.val_dice.compute(),
            'iou/validation': self.val_iou.compute(),
            'specificity/validation': self.val_specificity.compute(),
            'sensitivity/validation': self.val_sensitivity.compute(),
        }

    def test(self):
        """Evaluate model on test set after training is complete"""
        if 'test_loader' not in self.config or self.config['test_loader'] is None:
            logger.warning("No test_loader provided in config, skipping test evaluation")
            return {}
        
        logger.info("Running final evaluation on test set...")
        self.config['model'].eval()
        total_loss = 0.0
        num_batches = 0
        self.test_accuracy.reset()
        
        # Make test progress bar visible
        self.progress.update(self.task_test, visible=True)
        
        # Storage for visualization
        sample_X, sample_y, sample_probs = None, None, None
        
        with t.no_grad():
            for X, y in self.config['test_loader']:
                X, y = X.to(settings.device), y.to(settings.device)
                logits = self.config['model'](X)
                loss = self.config['loss_function'](logits, y)
                
                # Update metrics
                probs = F.sigmoid(logits) 
                self.test_accuracy.update(probs, y, threshold=threshold)
                self.test_dice.update(probs, y, threshold=threshold)
                self.test_iou.update(probs, y, threshold=threshold)
                self.test_sensitivity.update(probs, y, threshold=threshold)
                self.test_specificity.update(probs, y, threshold=threshold)
                total_loss += loss.item()
                num_batches += 1
                
                # Save first batch for visualization
                if sample_X is None:
                    sample_X = X
                    sample_y = y
                    sample_probs = probs
                
                self.progress.update(self.task_test, advance=1)
        
        # Plot one sample from test set
        if sample_X is not None:
            logger.info("Saving test set visualization...")
            self._plot_test_sample(sample_X, sample_y, sample_probs)
        
        logger.info("Test evaluation complete")
        
        return {
            'loss/test': total_loss / num_batches,
            'accuracy/test': self.test_accuracy.compute(),
            'dice/test': self.test_dice.compute(),
            'iou/test': self.test_iou.compute(),
            'specificity/test': self.test_specificity.compute(),
            'sensitivity/test': self.test_sensitivity.compute(),
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
            
            # Run test evaluation after all training epochs
            test_results = self.test()
            if test_results:
                self.experiment.log(test_results)
                logger.info(f"Test Results: {test_results}")

        except Exception as e: # TODO: 
            logger.error(f"Experiment run failed with error: {e}")
        
        finally:
            self.progress.stop()
            self.experiment.finish()

    def _plot_test_sample(self, X, y, probs):
        """Plot a single test sample prediction at the end of training"""
        # Convert tensors to numpy for plotting
        X_np = X[0].detach().cpu().numpy()  # Take first sample
        y_np = y[0].detach().cpu().numpy().squeeze()  # Assuming y has shape [batch, 1, H, W]
        pred_np = (probs[0].detach().cpu().numpy() > threshold).astype(np.float32)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input image
        img = X_np.transpose(1, 2, 0)
        if img.shape[2] == 1:  # Grayscale
            img = img.squeeze()
            axes[0].imshow(img, cmap='gray')
        else:  # RGB
            # Denormalize if needed
            if img.min() < 0 or img.max() > 1:
                img = (img - img.min()) / (img.max() - img.min())
            axes[0].imshow(img)
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
        
        # True mask
        axes[1].imshow(y_np, cmap='gray')
        axes[1].set_title('Ground Truth Mask', fontsize=14)
        axes[1].axis('off')
        
        # Prediction
        pred_mask = pred_np[0] if pred_np.ndim == 3 else pred_np
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Predicted Mask', fontsize=14)
        axes[2].axis('off')
        
        plt.suptitle('Test Set Segmentation Result', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save to results folder
        import os
        os.makedirs('./results', exist_ok=True)
        plt.savefig('./results/test_prediction.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Test visualization saved to ./results/test_prediction.png")