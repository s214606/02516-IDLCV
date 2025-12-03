import torch as t
import wandb
import os
import numpy as np
from utilities.logger import get_logger
from config import settings
from rich.progress import Progress

logger = get_logger(__name__) 

class Experiment:
    def __init__(self, project_name: str, name: str, config: dict):
        self.project_name = project_name
        self.name = name
        self.config = config
        
        # Initialize Progress Bar
        self.progress = Progress()
        
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
        
        test_len = len(self.config['test_loader']) if 'test_loader' in self.config and self.config['test_loader'] else 0
        self.task_test = self.progress.add_task(
            "[yellow]Testing on test set...",
            total=test_len,
            visible=False
        )

    def save_checkpoint(self, epoch):
        """
        Saves only the current state as the 'last' checkpoint.
        """
        os.makedirs('checkpoints', exist_ok=True)
        state = {
            'epoch': epoch,
            'model_state_dict': self.config['model'].state_dict(),
            'optimizer_state_dict': self.config['optimizer'].state_dict(),
        }
        # Save as _last.pth
        t.save(state, f'checkpoints/{self.name}_last.pth')
        logger.info(f"Saved final model at epoch {epoch}")

    def _parse_config(self):
        return {k: str(v) for k, v in self.config.items() 
                if k not in ['train_loader', 'val_loader', 'test_loader', 'dataset', 'model', 'loss_function']}

    def _calculate_accuracy(self, scores, labels):
        # scores: (Batch, Num_Classes) -> Get index of max score
        _, preds = t.max(scores, 1)
        correct = (preds == labels).sum().item()
        return correct, preds

    def train(self, epoch):
        self.config['model'].train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch in self.config['train_loader']:
            images = batch['image'].to(settings.device)
            labels = batch['label'].to(settings.device)
            target_deltas = batch['bbox_target'].to(settings.device)
            
            self.config['optimizer'].zero_grad()
            
            cls_scores, bbox_deltas_pred = self.config['model'](images)
            
            loss = self.config['loss_function'](
                cls_scores, 
                bbox_deltas_pred, 
                labels, 
                target_deltas
            )
            
            loss.backward()
            self.config['optimizer'].step()
            
            # --- Metrics ---
            total_loss += loss.item()
            correct, _ = self._calculate_accuracy(cls_scores, labels)
            total_correct += correct
            total_samples += labels.size(0)
            
            num_batches += 1
            self.progress.update(self.task_train, advance=1)
        
        if self.config.get('scheduler'):
            self.config['scheduler'].step()

        return {
            'loss/train': total_loss / num_batches,
            'accuracy/train': total_correct / total_samples
        }

    def eval(self, epoch):
        self.config['model'].eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        # For visualization
        log_images = []
        visualize_this_epoch = (epoch % 25 == 0) or (epoch == 1)
        
        with t.no_grad():
            for i, batch in enumerate(self.config['val_loader']):
                images = batch['image'].to(settings.device)
                labels = batch['label'].to(settings.device)
                target_deltas = batch['bbox_target'].to(settings.device)

                cls_scores, bbox_deltas_pred = self.config['model'](images)
                
                loss = self.config['loss_function'](
                    cls_scores, 
                    bbox_deltas_pred, 
                    labels, 
                    target_deltas
                )
                
                # --- Metrics ---
                total_loss += loss.item()
                correct, preds = self._calculate_accuracy(cls_scores, labels)
                total_correct += correct
                total_samples += labels.size(0)
                num_batches += 1
                
                # --- Visualization (First batch only) ---
                if visualize_this_epoch and i == 0:
                    limit = min(len(images), 8)
                    for k in range(limit):
                        img_tensor = images[k].cpu()
                        pred_lbl = preds[k].item()
                        true_lbl = labels[k].item()
                        
                        pred_text = "Pothole" if pred_lbl == 1 else "Background"
                        true_text = "Pothole" if true_lbl == 1 else "Background"
                        
                        log_images.append(
                            wandb.Image(
                                img_tensor, 
                                caption=f"Pred: {pred_text}\nTrue: {true_text}"
                            )
                        )
                
                self.progress.update(self.task_val, advance=1)
        
        metrics = {
            'loss/validation': total_loss / num_batches,
            'accuracy/validation': total_correct / total_samples
        }
        
        if log_images:
            metrics['examples/validation_predictions'] = log_images
            
        return metrics

    def test(self):
        if 'test_loader' not in self.config or not self.config['test_loader']:
            return None

        self.progress.update(self.task_test, visible=True)
        self.config['model'].eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        with t.no_grad():
            for batch in self.config['test_loader']:
                images = batch['image'].to(settings.device)
                labels = batch['label'].to(settings.device)
                target_deltas = batch['bbox_target'].to(settings.device)

                cls_scores, bbox_deltas_pred = self.config['model'](images)
                
                loss = self.config['loss_function'](
                    cls_scores, 
                    bbox_deltas_pred, 
                    labels, 
                    target_deltas
                )
                
                total_loss += loss.item()
                correct, _ = self._calculate_accuracy(cls_scores, labels)
                total_correct += correct
                total_samples += labels.size(0)
                num_batches += 1
                self.progress.update(self.task_test, advance=1)

        return {
            'loss/test': total_loss / num_batches,
            'accuracy/test': total_correct / total_samples
        }

    def run(self):
        logger.info("Initializing Weights & Biases run")
        self.experiment = wandb.init(
            entity='IDLCV',
            project=self.project_name,
            name=self.name,
            config=self._parse_config()
        )

        logger.info("Starting experiment")
        self.progress.start()
        self.config['model'].to(settings.device)
        
        # NOTE: Removed best_val_loss tracking as we only save the final epoch

        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_results = self.train(epoch)
                test_results = self.eval(epoch)

                self.experiment.log(train_results | test_results)
                
                # NOTE: Removed intermediate checkpoint saving here
                
                self.progress.reset(self.task_train)
                self.progress.reset(self.task_val)
                self.progress.update(self.task, advance=1)
            
            # --- Save Final Checkpoint ---
            self.save_checkpoint(self.config['epochs'])

            test_results = self.test()
            if test_results:
                self.experiment.log(test_results)
                logger.info(f"Test Results: {test_results}")

        except Exception as e: 
            logger.error(f"Experiment run failed with error: {e}")
            raise e
        
        finally:
            self.progress.stop()
            self.experiment.finish()