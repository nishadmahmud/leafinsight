import os
import logging
from typing import Dict, Optional
from ultralytics import YOLO
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton for loading and managing YOLO models in memory.
    """
    _instance = None
    _models: Dict[str, YOLO] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def load_models(self):
        """
        Load all YOLO models from the weights directory into memory.
        Automatically scans for any .pt files in the weights folder.
        """
        logger.info("Initializing ModelManager and loading models...")
        self._models.clear()
        
        if not settings.WEIGHTS_DIR.exists():
            logger.warning(f"Weights directory {settings.WEIGHTS_DIR} does not exist.")
            return

        for weight_path in settings.WEIGHTS_DIR.glob("*.pt"):
            model_name = weight_path.stem
            try:
                # Load model on CPU by default to save resources
                model = YOLO(str(weight_path))
                self._models[model_name] = model
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")

    def get_model(self, model_name: str) -> Optional[YOLO]:
        """
        Retrieve a loaded model by name.
        """
        return self._models.get(model_name)

    def get_loaded_models(self) -> list:
        """
        Return a list of currently loaded model names.
        """
        return list(self._models.keys())

# Global singleton instance
model_manager = ModelManager()
