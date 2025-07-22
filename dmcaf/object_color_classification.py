from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os
import json
from .object_detection import ObjectDetector
from .util import crop_image, remove_image_background

class ObjectColorClassifier:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.object_detector = ObjectDetector()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, '..', 'assets', 'colors.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.colors = list(data['colors'])
        
    def classify_color(self, image_path, object_name='object'):
        # Detect objects in the image
        detections = self.object_detector.detect_objects_with_boxes(image_path, target_object=object_name)
        if not detections:
            print(f"No objects of type '{object_name}' detected.")
            return None, 0.0
        # Use the object with the highest confidence (most prominent object)
        best_detection = max(detections, key=lambda x: x['confidence'])
        crop_image(input_path=image_path, box=best_detection['box'], output_path=f"cropped_{image_path}")
        remove_image_background(input_path=f"cropped_{image_path}", output_path=f"bg_removed_{image_path}")

        # Load the processed image as a PIL Image
        processed_image = Image.open(f"bg_removed_{image_path}")

        color_texts = [f"a photo of a {color} {object_name}" for color in self.colors]
        inputs = self.processor(
            text=color_texts,
            images=processed_image,
            return_tensors="pt",
            padding=True
        )
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        best_idx = probs.argmax(dim=1).item()
        return self.colors[best_idx], probs[0, best_idx].item()

# Example usage:
if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save("example_image.png")
    classifier = ObjectColorClassifier()
    color, confidence = classifier.classify_color("example_image.png", "cat")
    print(f"Predicted color: {color} with confidence: {confidence:.2f}")

