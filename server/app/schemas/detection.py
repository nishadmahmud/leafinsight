from pydantic import BaseModel
from typing import List, Optional

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    model_name: str
    inference_time_ms: float
    detections: List[DetectionResult]
    image_base64: Optional[str] = None

class ComparisonResponse(BaseModel):
    results: List[DetectionResponse]
