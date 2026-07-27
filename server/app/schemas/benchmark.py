from pydantic import BaseModel
from typing import List

class BenchmarkResult(BaseModel):
    model_name: str
    inference_time_ms: float
    cpu_percent: float
    ram_usage_mb: float

class BenchmarkResponse(BaseModel):
    results: List[BenchmarkResult]
