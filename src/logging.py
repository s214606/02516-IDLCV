import wandb
import logging
from rich.logging import RichHandler
from dataclasses import dataclass


@dataclass
class WANDConfig:
    project_name: str = "IDLCV_Project"
    entity: str = "IDLCV"
    log_level: int = logging.DEBUG
    config: dict = None


class Logger:
    def __init__(self, config: WANDConfig):
        self.logger = logging.getLogger('IDLCV_Logger')
        self.logger.setLevel(logging.DEBUG)

        logging.basicConfig(
        level=config.log_level,
        handlers=[
            logging.FileHandler('logs/train.log', mode='a'),
            RichHandler(rich_tracebacks=True,show_path=True)
            ]
        )

        self.run = wandb.init(
            project=config.project_name,
            entity=config.entity,
            config=config.config
        )
