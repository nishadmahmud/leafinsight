from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

from app.core.model_manager import model_manager

@router.get("/models")
def get_models():
    """
    Returns the list of available models dynamically loaded from weights folder.
    """
    return {
        "available_models": model_manager.get_loaded_models()
    }
