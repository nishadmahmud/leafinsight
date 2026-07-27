from pydantic import BaseModel

class ExplainResponse(BaseModel):
    model_name: str
    method: str
    heatmap_base64: str
    overlay_base64: str
    inference_time_ms: float
