from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "PlantVision AI"
    API_V1_STR: str = "/api"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure directories exist
settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
