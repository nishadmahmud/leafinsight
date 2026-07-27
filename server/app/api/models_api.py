from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/models")
def get_models():
    """
    Returns the list of available models for detection.
    """
    return {
        "available_models": settings.AVAILABLE_MODELS
    }
