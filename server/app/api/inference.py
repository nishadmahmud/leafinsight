from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Optional
from app.services.detection import DetectionService
from app.services.comparison import ComparisonService
from app.services.explainability import ExplainabilityService
from app.services.benchmark import BenchmarkService
from app.schemas.detection import DetectionResponse, ComparisonResponse
from app.schemas.explainability import ExplainResponse
from app.schemas.benchmark import BenchmarkResponse
from app.core.config import settings

router = APIRouter()

@router.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(...),
    model_name: str = Form(...),
    conf_thresh: float = Form(0.25),
    iou_thresh: float = Form(0.45)
):
    """
    Run detection using a single selected YOLO model.
    """
    if model_name not in settings.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {model_name} is not supported.")
        
    image_bytes = await image.read()
    try:
        return DetectionService.run_inference(model_name, image_bytes, conf_thresh, iou_thresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare", response_model=ComparisonResponse)
async def compare(
    image: UploadFile = File(...),
    models: str = Form(...), # Comma separated list like 'yolov8s,yolov9s'
    conf_thresh: float = Form(0.25),
    iou_thresh: float = Form(0.45)
):
    """
    Run detection on multiple models simultaneously.
    """
    model_names = [m.strip() for m in models.split(',') if m.strip() in settings.AVAILABLE_MODELS]
    if not model_names:
        raise HTTPException(status_code=400, detail="No valid models selected for comparison.")
        
    image_bytes = await image.read()
    try:
        return ComparisonService.run_comparison(model_names, image_bytes, conf_thresh, iou_thresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain", response_model=ExplainResponse)
async def explain(
    image: UploadFile = File(...),
    model_name: str = Form(...),
    method: str = Form(...) # 'rise' or 'occlusion'
):
    """
    Generate an Explainable AI heatmap for the detected bounding box.
    """
    if model_name not in settings.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {model_name} is not supported.")
        
    image_bytes = await image.read()
    try:
        return ExplainabilityService.generate_explanation(model_name, method, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(
    image: UploadFile = File(...),
    models: str = Form(...)
):
    """
    Benchmark memory, CPU, and inference latency for selected models.
    """
    model_names = [m.strip() for m in models.split(',') if m.strip() in settings.AVAILABLE_MODELS]
    if not model_names:
        raise HTTPException(status_code=400, detail="No valid models selected for benchmarking.")
        
    image_bytes = await image.read()
    try:
        return BenchmarkService.run_benchmark(model_names, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
