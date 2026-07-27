import concurrent.futures
from typing import List
from app.schemas.detection import DetectionResponse, ComparisonResponse
from app.services.detection import DetectionService

class ComparisonService:
    @classmethod
    def run_comparison(cls, model_names: List[str], image_bytes: bytes, conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> ComparisonResponse:
        """
        Run inference on multiple models simultaneously using a ThreadPoolExecutor.
        """
        results = []
        
        # Use ThreadPoolExecutor for concurrent inference
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_names)) as executor:
            future_to_model = {
                executor.submit(
                    DetectionService.run_inference, 
                    model_name, 
                    image_bytes, 
                    conf_thresh, 
                    iou_thresh
                ): model_name for model_name in model_names
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    # In a real app, you might want to return an error response for that specific model
                    # For now, we will skip it or log it
                    print(f'{model_name} generated an exception: {exc}')
                    
        # Sort results to match original requested order
        sorted_results = []
        for requested_model in model_names:
            for r in results:
                if r.model_name == requested_model:
                    sorted_results.append(r)
                    break
                    
        return ComparisonResponse(results=sorted_results)
