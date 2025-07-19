import ultralytics
from ultralytics import YOLO
import cv2
from typing import Optional, List, Dict, Any
from PIL import Image
import requests
import numpy as np
from object_count import ObjectCounter

class ObjectSegmenter:
    def __init__(self, model_path='yolov8n-seg.pt'):
        """
        Initializes the ObjectSegmenter with a YOLO segmentation model.
        """
        self.model = YOLO(model_path)

    def segment_objects_in_image(self, image_path: str, target_object: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs instance segmentation on an image.
        Returns a list of dicts with class name, bounding box, and mask.
        """
        if cv2.imread(image_path) is None:
            print(f"Warning: Image at path {image_path} could not be loaded.")
            return []
        
        results = self.model.predict(image_path, verbose=False)
        result = results[0]
        
        segments = []
        for box, mask in zip(result.boxes, result.masks.data if result.masks is not None else []):
            class_name = self.model.names[int(box.cls)]
            if target_object and class_name != target_object:
                continue
            coordinates = box.xyxy[0].tolist()
            mask_array = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
            segments.append({
                'class_name': class_name,
                'box': coordinates,
                'mask': mask_array
            })
        return segments

    def visualize_and_save(self, image_path: str, segments: List[Dict[str, Any]], output_path: str = "segmentation_result.png"):
        """
        Visualizes segmentation masks and saves the result as an image.
        """
        image = cv2.imread(image_path)
        overlay = image.copy()
        for seg in segments:
            mask = seg['mask']
            # Resize mask to image size if needed
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            overlay[mask_resized > 0.5] = color
        result_img = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
        cv2.imwrite(output_path, result_img)
        print(f"Segmentation visualization saved to {output_path}")

    def remove_background(self, image_path: str, segments: List[Dict[str, Any]], output_path: str = "foreground.png"):
        """
        Removes the background using segmentation masks and saves the foreground.
        """
        image = cv2.imread(image_path)
        mask_total = np.zeros(image.shape[:2], dtype=np.uint8)
        for seg in segments:
            mask = seg['mask']
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask_total = np.logical_or(mask_total, mask_resized > 0.5)
        # Create an alpha channel
        foreground = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        foreground[..., 3] = mask_total.astype(np.uint8) * 255
        cv2.imwrite(output_path, foreground)
        print(f"Foreground saved to {output_path}")

if __name__ == "__main__":
    # Get an image from the COCO dataset
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # Save the image locally for cv2 and YOLO
    image_path = "coco_sample.jpg"
    image.save(image_path)
    segmenter = ObjectSegmenter()

    remove_background = True  

    if remove_background:
        # call remove_background method

        segments = segmenter.segment_objects_in_image(image_path)
        segmenter.remove_background(image_path, segments, output_path="foreground.png")
    else:
        segments = segmenter.segment_objects_in_image(image_path)
        print(f"Number of segmented objects: {len(segments)}")
        if segments:
            print(f"First segment info: {segments[0]}")
            segmenter.visualize_and_save(image_path, segments, output_path="coco_segmentation_result.png")
