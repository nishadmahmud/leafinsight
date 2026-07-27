import psutil
from typing import List
from app.schemas.benchmark import BenchmarkResponse, BenchmarkResult
from app.services.detection import DetectionService
import concurrent.futures

class BenchmarkService:
    @classmethod
    def run_benchmark(cls, model_names: List[str], image_bytes: bytes) -> BenchmarkResponse:
        results = []
        process = psutil.Process()
        
        # We benchmark sequentially to get accurate resource readings per model
        for model_name in model_names:
            # Baseline usage
            psutil.cpu_percent(interval=0.1)
            base_mem = process.memory_info().rss
            
            # Run inference multiple times for average? For now single run is fine
            det_result = DetectionService.run_inference(model_name, image_bytes)
            
            # Post usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            mem_usage = process.memory_info().rss
            mem_diff = max(0, (mem_usage - base_mem) / (1024 * 1024)) # MB
            
            # Total mem used right now
            current_mem_mb = process.memory_info().rss / (1024 * 1024)
            
            results.append(BenchmarkResult(
                model_name=model_name,
                inference_time_ms=det_result.inference_time_ms,
                cpu_percent=cpu_usage,
                ram_usage_mb=current_mem_mb
            ))
            
        return BenchmarkResponse(results=results)
