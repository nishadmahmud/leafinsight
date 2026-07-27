from fastapi import APIRouter
import psutil
from app.core.model_manager import model_manager

router = APIRouter()

@router.get("/health")
def get_health():
    """
    Returns the server status, loaded models, and memory usage.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "status": "healthy",
        "loaded_models": model_manager.get_loaded_models(),
        "memory_usage_mb": round(memory_info.rss / (1024 * 1024), 2)
    }
