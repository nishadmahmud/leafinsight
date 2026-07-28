import time
import base64
import cv2
import numpy as np
from PIL import Image
import io
from app.core.model_manager import model_manager
from app.schemas.detection import DetectionResponse, DetectionResult, BoundingBox

class DetectionService:
    @staticmethod
    def process_image(image_bytes: bytes) -> np.ndarray:
        """Convert uploaded image bytes to OpenCV format (BGR)."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def encode_image_base64(image: np.ndarray) -> str:
        """Encode OpenCV image to base64 string."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def draw_boxes(image: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        img_copy = image.copy()
        for det in detections:
            bbox = det.bbox
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det.class_name} ({det.confidence:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Ensure label doesn't get cut off at the right edge
            label_x1 = x1
            img_w = img_copy.shape[1]
            if label_x1 + w > img_w:
                label_x1 = max(0, img_w - w)
                
            # Ensure label doesn't get cut off at the top of the image
            if y1 - 20 < 0:
                label_y1, label_y2 = y1, y1 + 20
                text_y = label_y2 - 5
            else:
                label_y1, label_y2 = y1 - 20, y1
                text_y = y1 - 5
                
            cv2.rectangle(img_copy, (label_x1, label_y1), (label_x1 + w, label_y2), (0, 255, 0), -1)
            cv2.putText(img_copy, label, (label_x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return img_copy

    @classmethod
    def run_inference(cls, model_name: str, image_bytes: bytes, conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> DetectionResponse:
        model = model_manager.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} is not loaded or available.")

        image = cls.process_image(image_bytes)
        
        start_time = time.time()
        # Run YOLO inference
        results = model(image, conf=conf_thresh, iou=iou_thresh, verbose=False)
        inference_time_ms = (time.time() - start_time) * 1000

        result = results[0]
        detections = []
        
        # Parse results
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            
            detections.append(DetectionResult(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            ))
            
        # Draw and encode image
        annotated_img = cls.draw_boxes(image, detections)
        img_base64 = cls.encode_image_base64(annotated_img)

        return DetectionResponse(
            model_name=model_name,
            inference_time_ms=inference_time_ms,
            detections=detections,
            image_base64=img_base64
        )
