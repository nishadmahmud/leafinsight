from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core.model_manager import model_manager
from app.api import health, models_api, inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI startup and shutdown events.
    Loads YOLO models into memory upon startup.
    """
    logger.info("Starting up PlantVision AI backend...")
    model_manager.load_models()
    yield
    logger.info("Shutting down PlantVision AI backend...")
    # Add any cleanup here if needed

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Comparative Explainable Deep Learning Platform for Indoor Plant Leaf Disease Detection",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (including Vercel deployed frontend)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}

# Include routers
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(models_api.router, prefix=settings.API_V1_STR)
app.include_router(inference.router, prefix=settings.API_V1_STR)
