import ultralytics
from ultralytics import YOLO
import cv2
from typing import Optional, List, Dict, Any

class ObjectCounter:
    def __init__(self, model_path='yolo11n.pt'):
        """
        Initializes the ObjectCounter with a YOLO model.
        """
        self.model = YOLO(model_path)

    def count_objects_in_image(self, image_path: str, output_path: str = None, target_object: Optional[str] = None) -> int:
        """
        Counts objects in an image, optionally filtering for a specific object.
        """
        if cv2.imread(image_path) is None:
            print(f"Warning: Image at path {image_path} could not be loaded.")
            return 0
        # Perform inference
        results = self.model.predict(image_path, verbose=False)

        print(f"Detected {len(results)} results for image {image_path}")
        result = results[0]

        if target_object:
            # Filter detections by class name
            count = 0
            for box in result.boxes:
                class_name = self.model.names[int(box.cls)]
                if class_name == target_object:
                    count += 1
            return count
        else:
            # Count all detected objects
            return len(result.boxes)
    
    def detect_objects_with_boxes(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detects objects in an image and returns their class names and bounding boxes.
        """
        if cv2.imread(image_path) is None:
            print(f"Warning: Image at path {image_path} could not be loaded.")
            return []
        
        results = self.model.predict(image_path, verbose=False)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            class_name = self.model.names[int(box.cls)]
            # Bounding box format [x1, y1, x2, y2]
            coordinates = box.xyxy[0].tolist()
            detections.append({'class_name': class_name, 'box': coordinates})
            
        return detections


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