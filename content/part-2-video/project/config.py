import logging
import torch as t

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    root_dir: str = 'data/ufc10'

    device: str = 'cuda' if t.cuda.is_available() else 'cpu'

    # seed: int = 42
    log_level: int = logging.INFO

    model_config = SettingsConfigDict() # optional prefix for environment variables

settings = Settings()

logger.info(f'Using device: {settings.device}')

settings = Settings()
