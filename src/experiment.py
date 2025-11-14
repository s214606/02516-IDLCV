import torch
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_function: object
    epochs: int = 50
    learning_rate: float = 1e-4

class Experiment:
    def __init__(
            self,
            name: str,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_function: object,
            epochs: int = 50,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.0,
            tags: list[str] = None,
            ):
        
        self.model = model
        self.optim = optimizer(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay
            
        )
        self.name = name
        self.tags = tags if tags is not None else []
    
    def train():
        raise NotImplementedError
    
    def eval():
        raise NotImplementedError
    
    def run():
        raise NotImplementedError

class ExperimentOrchestrator:
    def __init__(
            self,
            entity: str = 'IDLCV',
            project_name: str = 'Untitled Project',
            ):
        
        self.experiments = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add_experiment(self, experiment: Experiment):
        self.experiments.append(experiment)
        return self
    
    def run(self):
        for experiment in self.experiments:
            print(f"Running experiment: {experiment.name} in project: {experiment.project_name}")
            # Here you would add the logic to train and evaluate the model
            # using the configuration provided in experiment.config
            experiment.run()