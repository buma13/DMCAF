from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, Dict, Any

class ObjectCounter:
    def __init__(self, model_path='yolo11m-seg.pt'):
        """
        Initializes the ObjectCounter with a YOLO segmentation model.
        """
        self.model = YOLO(model_path)

    def count_objects_in_image(self, image_path: str, target_object: Optional[str] = None,
                              target_class_id: Optional[int] = None, conf_threshold: float = 0.4) -> int:
        """
        Counts objects in an image using segmentation, optionally filtering for a specific object.
        """
        result = self.count_and_segment_objects(image_path, target_object, target_class_id, conf_threshold)
        return result['count']
    
    def count_and_segment_objects(self, image_path: str, target_object: Optional[str] = None, 
                                 target_class_id: Optional[int] = None, conf_threshold: float = 0.4) -> Dict[str, Any]:
        """
        Counts objects and returns segmentation data including masks and boxes.
        
        Args:
            image_path: Path to the image file
            target_object: Target object name for filtering (deprecated, use target_class_id)
            target_class_id: YOLO class ID for filtering detections
            conf_threshold: Confidence threshold for detections
        """
        if cv2.imread(image_path) is None:
            print(f"Warning: Image at path {image_path} could not be loaded.")
            return {'count': 0, 'detections': [], 'image_shape': None}
        
        # Prepare prediction arguments
        predict_kwargs = {
            'verbose': False, 
            'save': True,
            'conf': conf_threshold,
            'project': "yolo_outputs",
            'name': 'count',
            'exist_ok': True
        }
        
        # Add class filtering if target_class_id is specified
        if target_class_id is not None:
            predict_kwargs['classes'] = [target_class_id]
        
        results = self.model.predict(image_path, **predict_kwargs)
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
            
            # If target_object specified, filter by it (backward compatibility)
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
    import requests
    from PIL import Image
    url = "https://farm9.staticflickr.com/8124/8687257944_378b3b87bd_z.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save("example_image_count.png")
    counter = ObjectCounter()
    # Example 1: Count all objects
    total_count = counter.count_objects_in_image("example_image_count.png")
    print(f"Total number of objects detected: {total_count}")
    # Example 2: Count specific objects
    specific_count = counter.count_objects_in_image("example_image_count.png", target_object='banana')
    print(f"Number of bananas detected: {specific_count}")