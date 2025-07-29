import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

class ObjectCounter:
    def __init__(self, model_path='yolo11m-seg.pt'):
        """
        Initializes the ObjectCounter with a YOLO segmentation model.
        """
        self.model = YOLO(model_path)

    def count_objects_in_image(self, image_path: str, target_object: Optional[str] = None) -> int:
        """
        Counts objects in an image using segmentation, optionally filtering for a specific object.
        """
        result = self.count_and_segment_objects(image_path, target_object)
        return result['count']
    
    def count_and_segment_objects(self, image_path: str, target_object: Optional[str] = None) -> Dict[str, Any]:
        """
        Counts objects and returns segmentation data including masks and boxes.
        """
        if cv2.imread(image_path) is None:
            print(f"Warning: Image at path {image_path} could not be loaded.")
            return {'count': 0, 'detections': [], 'image_shape': None}
        
        results = self.model.predict(image_path, verbose=False, save=True)
        result = results[0]
        
        # Get image dimensions
        image = cv2.imread(image_path)
        image_shape = image.shape[:2]  # (height, width)

        # Use masks instead of boxes for counting
        if result.masks is None:
            return {'count': 0, 'detections': [], 'image_shape': image_shape}

        detections = []
        count = 0
        
        for box, mask in zip(result.boxes, result.masks.data):
            class_name = self.model.names[int(box.cls)]
            
            # If target_object specified, filter by it
            if target_object and class_name != target_object:
                continue
                
            count += 1
            
            # Get box coordinates and confidence
            coordinates = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            
            # Get mask array
            mask_array = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
            
            detections.append({
                'class_name': class_name,
                'box': coordinates,
                'confidence': confidence,
                'mask': mask_array
            })
        
        return {
            'count': count,
            'detections': detections,
            'image_shape': image_shape
        }


__main__ = "__main__"
if __name__ == "__main__":
    # Example usage
    image_path = 'to_be_defined/experiment_000/experiment_000_stable-diffusion-v1-5_stable-diffusion-v1-5_gs7.5_steps50_cond7.png' # Please update this path
    # model_path = 'yolov8n.pt' # Optional: Path to your YOLO model
    counter = ObjectCounter()
    # Example 1: Count all objects
    total_count = counter.count_objects_in_image(image_path)
    print(f"Total number of objects detected: {total_count}")
    # Example 2: Count specific objects
    specific_count = counter.count_objects_in_image(image_path, target_object='children')
    print(f"Number of cats detected: {specific_count}")