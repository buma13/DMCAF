
from ultralytics import YOLO
import cv2
from typing import Optional, List, Dict, Any

class ObjectDetector:
    def __init__(self, model_path='yolo11m.pt'):
        """
        Initializes the ObjectDetector with a YOLO model.
        """
        self.model = YOLO(model_path)
    
    def detect_objects_with_boxes(self, image_path: str, target_object: str = None, 
                                 target_class_id: Optional[int] = None, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detects objects in an image and returns their class names and bounding boxes.
        
        Args:
            image_path: Path to the image file
            target_object: Target object name for filtering (deprecated, use target_class_id)
            target_class_id: YOLO class ID for filtering detections
            conf_threshold: Confidence threshold for detections
        """
        if cv2.imread(image_path) is None:
            print(f"Warning: Image at path {image_path} could not be loaded.")
            return []
        
        # Prepare prediction arguments
        predict_kwargs = {
            'verbose': False, 
            'save': True,
            'conf': conf_threshold
        }
        
        # Add class filtering if target_class_id is specified
        if target_class_id is not None:
            predict_kwargs['classes'] = [target_class_id]
        
        results = self.model.predict(image_path, **predict_kwargs)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            class_name = self.model.names[int(box.cls)]
            if target_object and class_name != target_object:
                continue
            # Bounding box format [x1, y1, x2, y2]
            coordinates = box.xyxy[0].tolist()
            confidence = float(box.conf[0])  # Get confidence score
            detections.append({
                'class_name': class_name,
                'box': coordinates,
                'confidence': confidence
            })
        
        return detections
