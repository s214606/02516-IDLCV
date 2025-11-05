import torch as t
import wandb
from utils.logger import get_logger
from config import settings
from typing import Dict, Any
from rich.progress import Progress
from metrics.classification import Accuracy

logger = get_logger(__name__) 

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
        self.val_accuracy = Accuracy()

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
            total=len(self.config['test_loader'])
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
            self.train_accuracy.update(logits, y)
            total_loss += loss.item()
            num_batches += 1
            
            self.progress.update(self.task_train, advance=1)
        
        if self.config['scheduler']:
            before_lr = self.config['optimizer'].param_groups[0]['lr']
            self.config['scheduler'].step()
            after_lr = self.config['optimizer'].param_groups[0]['lr']

            if before_lr != after_lr:
                logger.info("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))


        return {
            'loss/train': total_loss / num_batches,
            'accuracy/train': self.train_accuracy.compute()
        }

    def eval(self, epoch):
        self.config['model'].eval()
        total_loss = 0.0
        num_batches = 0
        self.val_accuracy.reset()  # Reset accuracy at start of validation
        
        with t.no_grad():
            for X, y in self.config['test_loader']:
                X, y = X.to(settings.device), y.to(settings.device)
                logits = self.config['model'](X)
                loss = self.config['loss_function'](logits, y)
                
                # Update metrics
                self.val_accuracy.update(logits, y)
                total_loss += loss.item()
                num_batches += 1
                
                self.progress.update(self.task_val, advance=1)
        
        return {
            'loss/validation': total_loss / num_batches,
            'accuracy/validation': self.val_accuracy.compute()
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
        
        logger.info("Moving model to GPU")
        self.config['model'].to(settings.device)

        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_results = self.train(epoch)
                test_results = self.eval(epoch)

                self.experiment.log(train_results | test_results)
                if 'scheduler' in self.config:
                    self.config['scheduler'].step()
                
                self.progress.reset(self.task_train)
                self.progress.reset(self.task_val)
                self.progress.update(self.task, advance=1)

        except Exception as e: # TODO: 
            logger.error(f"Experiment run failed with error: {e}")
        
        finally:
            self.progress.stop()
            self.experiment.finish()

class TwoStreamFusion:
    """
    Two-stream fusion experiment for combining pre-trained spatial and temporal models.
    """
    def __init__(self, project_name: str, name: str, config: dict):
        self.project_name = project_name
        self.name = name
        self.config = config
        frame_loader_len = len(self.config['frame_test_loader'])
        flow_loader_len = len(self.config['flow_test_loader'])

        # Get a sample batch to check shapes
        frame_batch = next(iter(self.config['frame_test_loader']))
        flow_batch = next(iter(self.config['flow_test_loader']))

        # Handle both cases: batch might be (data, labels) tuple or just data
        frame_shape = frame_batch[0].shape if isinstance(frame_batch, (tuple, list)) else frame_batch.shape
        flow_shape = flow_batch[0].shape if isinstance(flow_batch, (tuple, list)) else flow_batch.shape

        print(f"Frame test loader - Length: {frame_loader_len}, Batch shape: {frame_shape}")
        print(f"Flow test loader  - Length: {flow_loader_len}, Batch shape: {flow_shape}")
        print(f"\nDiagnostics:")
        print(f"  Frame dataset size: {len(self.config['frame_test_loader'].dataset)}")
        print(f"  Flow dataset size: {len(self.config['flow_test_loader'].dataset)}")
        print(f"  Frame batch size: {self.config['frame_test_loader'].batch_size}")
        print(f"  Flow batch size: {self.config['flow_test_loader'].batch_size}")
        print(f"\nLength ratio: {flow_loader_len / frame_loader_len:.2f}x")

        assert len(self.config['frame_test_loader']) == len(self.config['flow_test_loader']), \
            "Frame and flow test loaders must have the same number of batches"
        
        self.progress = Progress()
        self.val_accuracy = Accuracy()
        
        self.task = self.progress.add_task(
            "[cyan]Evaluating two-stream fusion...",
            total=len(self.config['frame_test_loader'])
        )
    
    def _extract_single_frame(self, frame_stack: t.Tensor) -> t.Tensor:
        """
        Extract a single random frame from the temporal stack.
        
        Args:
            frame_stack: Tensor of shape [batch, channels, temporal, height, width]
        
        Returns:
            Single frame tensor of shape [batch, channels, height, width]
        """
        # frame_stack shape: [batch, 3, 10, 64, 64]
        temporal_dim = frame_stack.shape[2]
        random_idx = t.randint(0, temporal_dim, (1,)).item()
        return frame_stack[:, :, random_idx, :, :]  # [batch, 3, 64, 64]
    
    def _fuse_predictions(self, frame_input: t.Tensor, flow_input: t.Tensor) -> t.Tensor:
        """
        Fuse predictions from spatial and temporal streams.
        
        Args:
            frame_input: Spatial (frame) input tensor
            flow_input: Temporal (optical flow) input tensor
        
        Returns:
            Fused probability predictions
        """
        # Extract single frame if input has temporal dimension
        if frame_input.dim() == 5:  # [batch, channels, temporal, height, width]
            frame_input = self._extract_single_frame(frame_input)
        
        # Get predictions from both streams
        spatial_logits = self.config['spatial_model'](frame_input)
        temporal_logits = self.config['temporal_model'](flow_input)
        
        # Apply softmax and average
        spatial_probs = t.softmax(spatial_logits, dim=1)
        temporal_probs = t.softmax(temporal_logits, dim=1)
        fused_probs = (spatial_probs*0.3 + temporal_probs*0.7)
        
        return fused_probs
    
    def eval(self):
        """Evaluate the fused two-stream model"""
        self.config['spatial_model'].eval()
        self.config['temporal_model'].eval()
        
        total_loss = 0.0
        num_batches = 0
        self.val_accuracy.reset()
        
        with t.no_grad():
            for (frame_X, frame_y), (flow_X, flow_y) in zip(
                self.config['frame_test_loader'],
                self.config['flow_test_loader']
            ):
                assert t.equal(frame_y, flow_y), "Labels must match between frame and flow loaders"
                
                frame_X = frame_X.to(settings.device)
                flow_X = flow_X.to(settings.device)
                y = frame_y.to(settings.device)
                
                # Use unified fusion logic (handles frame extraction internally)
                fused_probs = self._fuse_predictions(frame_X, flow_X)
                fused_logits = t.log(fused_probs + 1e-10)
                
                self.val_accuracy.update(fused_logits, y)
                
                if 'loss_function' in self.config:
                    loss = self.config['loss_function'](fused_logits, y)
                    total_loss += loss.item()
                
                num_batches += 1
                self.progress.update(self.task, advance=1)
        
        results = {'accuracy/fusion': self.val_accuracy.compute()}
        if 'loss_function' in self.config:
            results['loss/fusion'] = total_loss / num_batches
            
        return results
    
    def predict(self, frame_input: t.Tensor, flow_input: t.Tensor) -> t.Tensor:
        """Make predictions using the fused two-stream model."""
        self.config['spatial_model'].eval()
        self.config['temporal_model'].eval()
        
        with t.no_grad():
            frame_input = frame_input.to(settings.device)
            flow_input = flow_input.to(settings.device)
            return self._fuse_predictions(frame_input, flow_input)
    
    def run(self):
        """Run the two-stream fusion evaluation"""
        logger.info("Initializing Weights & Biases run for two-stream fusion")
        self.experiment = wandb.init(
            entity='IDLCV',
            project=self.project_name,
            name=self.name,
            config=self.config
        )
        
        logger.info("Starting two-stream fusion evaluation")
        self.progress.start()
        
        logger.info("Moving models to GPU")
        self.config['spatial_model'].to(settings.device)
        self.config['temporal_model'].to(settings.device)
        
        try:
            results = self.eval()
            self.experiment.log(results)
            logger.info(f"Fusion results: {results}")
            
        except Exception as e:
            logger.error(f"Two-stream fusion evaluation failed with error: {e}")
        
        finally:
            self.progress.stop()
            self.experiment.finish()

