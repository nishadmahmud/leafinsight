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
        If a weights file is missing, it skips it (useful for initial dev).
        """
        logger.info("Initializing ModelManager and loading models...")
        for model_name in settings.AVAILABLE_MODELS:
            weight_path = settings.WEIGHTS_DIR / f"{model_name}.pt"
            if weight_path.exists():
                try:
                    # Load model on CPU by default to save resources
                    model = YOLO(str(weight_path))
                    # Optionally force CPU if needed: model.to('cpu')
                    self._models[model_name] = model
                    logger.info(f"Successfully loaded {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
            else:
                logger.warning(f"Weights file not found for {model_name} at {weight_path}. Model not loaded.")

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
