
from ultralytics import YOLO
import cv2
from typing import Optional, List, Dict, Any

class ObjectDetector:
    def __init__(self, model_path='yolo11n.pt'):
        """
        Initializes the ObjectDetector with a YOLO model.
        """
        self.model = YOLO(model_path)
    
    def detect_objects_with_boxes(self, image_path: str, target_object: str=None) -> List[Dict[str, Any]]:
        """
        Detects objects in an image and returns their class names and bounding boxes.
        """
        if cv2.imread(image_path) is None:
            print(f"Warning: Image at path {image_path} could not be loaded.")
            return []
        
        results = self.model.predict(image_path, verbose=False, save=True)
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
