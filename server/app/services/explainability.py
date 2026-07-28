import time
import cv2
import numpy as np
import base64
from typing import List
from app.core.model_manager import model_manager
from app.schemas.explainability import ExplainResponse
from app.services.detection import DetectionService

class ExplainabilityService:
    @staticmethod
    def _apply_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Apply a jet colormap to the heatmap and blend with original image."""
        heatmap = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # overlay
        overlay = cv2.addWeighted(image, 0.5, colored_heatmap, 0.5, 0)
        return colored_heatmap, overlay

    @classmethod
    def _run_occlusion(cls, model, image: np.ndarray, class_id: int, base_conf: float, box_size: int = 10, step: int = 10) -> np.ndarray:
        """
        Slide a gray box across the image and record the drop in confidence for the target class.
        """
        h, w, _ = image.shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h, step):
            for x in range(0, w, step):
                # create occluded image
                occluded_img = image.copy()
                y2, x2 = min(y + box_size, h), min(x + box_size, w)
                occluded_img[y:y2, x:x2] = (128, 128, 128)

                # run inference
                results = model(occluded_img, verbose=False)
                max_conf = 0.0
                
                # Check confidence of target class
                for box in results[0].boxes:
                    if int(box.cls[0].item()) == class_id:
                        max_conf = max(max_conf, box.conf[0].item())
                
                # The drop in confidence represents importance
                importance = max(0.0, base_conf - max_conf)
                heatmap[y:y2, x:x2] += importance

        # Normalize
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

    @classmethod
    def _run_rise(cls, model, image: np.ndarray, class_id: int, N: int = 100, s: int = 8, p1: float = 0.5) -> np.ndarray:
        """
        Randomized Input Sampling for Explanation (RISE) adapted for object detection.
        """
        h, w, _ = image.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # We need smaller resolution masks upsampled
        cell_size = np.ceil(np.array([h, w]) / s)
        up_size = (int((s + 1) * cell_size[0]), int((s + 1) * cell_size[1]))

        for i in range(N):
            # Generate random binary mask
            mask = np.random.rand(s, s) < p1
            mask = mask.astype(np.float32)
            
            # Upsample (INTER_NEAREST creates the blocky grid effect seen in the paper)
            up_mask = cv2.resize(mask, (up_size[1], up_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # No random shift (to preserve the perfect 8x8 grid seen in the paper)
            shift_x = 0
            shift_y = 0
            
            crop_mask = up_mask[shift_y:shift_y + h, shift_x:shift_x + w]
            
            # Apply mask to image
            masked_img = image.copy()
            for c in range(3):
                masked_img[:, :, c] = masked_img[:, :, c] * crop_mask
            
            masked_img = masked_img.astype(np.uint8)
            
            # Run inference
            results = model(masked_img, verbose=False)
            max_conf = 0.0
            for box in results[0].boxes:
                if int(box.cls[0].item()) == class_id:
                    max_conf = max(max_conf, box.conf[0].item())
                    
            heatmap += crop_mask * max_conf
            
        # Normalize
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

    @classmethod
    def generate_explanation(cls, model_name: str, method: str, image_bytes: bytes) -> ExplainResponse:
        model = model_manager.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} is not loaded.")

        image = DetectionService.process_image(image_bytes)
        
        start_time = time.time()
        
        # First find the primary detection to explain
        base_results = model(image, verbose=False)
        if len(base_results[0].boxes) == 0:
            raise ValueError("No objects detected to explain.")
            
        # We explain the most confident detection
        best_box = None
        best_conf = 0.0
        
        for box in base_results[0].boxes:
            conf = box.conf[0].item()
            if conf > best_conf:
                best_conf = conf
                best_box = box
                
        target_class_id = int(best_box.cls[0].item())
        
        if method.lower() == 'occlusion':
            heatmap = cls._run_occlusion(model, image, target_class_id, best_conf)
        elif method.lower() == 'rise':
            heatmap = cls._run_rise(model, image, target_class_id, N=50) # Reduced N for speed
        else:
            raise ValueError("Unsupported method. Choose 'occlusion' or 'rise'.")

        inference_time_ms = (time.time() - start_time) * 1000
        
        # Apply colormap
        heatmap_img, overlay_img = cls._apply_heatmap(image, heatmap)
        
        # Base64 encode
        heatmap_b64 = DetectionService.encode_image_base64(heatmap_img)
        overlay_b64 = DetectionService.encode_image_base64(overlay_img)
        
        return ExplainResponse(
            model_name=model_name,
            method=method,
            heatmap_base64=heatmap_b64,
            overlay_base64=overlay_b64,
            inference_time_ms=inference_time_ms
        )
