from pydantic_settings import BaseSettings
from collections.abc import Sequence

class DataSettings(BaseSettings):
    data_dir: str = '/dtu/datasets1/02516/'
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True

class ExperimentSettings(BaseSettings):
    name: str = 'Untitled Experiment'
    config: dict = []
    tags: Sequence[str] = []

class ProjectSettings(BaseSettings):
    entity: str = 'IDLCV' # This should not change unless team at WANDB is changed
    name: str = 'Untitled Project'



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def to_wandb_config(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
