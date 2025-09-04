from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os
import json
from .object_detection import ObjectDetector
from .object_segmentation import ObjectSegmenter
from .util import crop_image, remove_image_background

from PIL import Image
import numpy as np
import json
from sklearn.cluster import KMeans

class ObjectColorClassifier:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.object_detector = ObjectDetector()
        self.object_segmenter = ObjectSegmenter()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, '..', 'assets', 'colors.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.color_data = data
            self.colors = [c['name'] for c in data['colors']]
        
    def classify_color(self, image_path, detections, object_name='object'):
        # Detect objects in the image
        #detections = self.object_detector.detect_objects_with_boxes(image_path, target_object=object_name)
        #detections = self.object_segmenter.segment_objects_in_image(image_path, target_object=object_name)

        if not detections:
            print(f"No objects of type '{object_name}' detected.")
            return None, 0.0
        # Use the object with the highest confidence (most prominent object)
        best_detection = max(detections, key=lambda x: x['confidence'])

        # Create correct paths for intermediate files
        img_dir = os.path.dirname(image_path)
        img_name = os.path.basename(image_path)
        
        # Create a subdirectory for processed images to avoid clutter
        processed_dir = os.path.join(img_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        cropped_path = os.path.join(processed_dir, f"cropped_{img_name}")
        bg_removed_path = os.path.join(processed_dir, f"bg_removed_{img_name}")

        #crop_image(input_path=image_path, box=best_detection['box'], output_path=cropped_path)
        #remove_image_background(input_path=cropped_path, output_path=bg_removed_path)
        self.object_segmenter.remove_background(image_path=image_path, segments=[best_detection], output_path=bg_removed_path)
        crop_image(input_path=bg_removed_path, box=best_detection['box'], output_path=cropped_path)

        # Load the processed image as a PIL Image
        processed_image = Image.open(cropped_path)

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


    def classify_color_knn(self, image_path, detections):
        """
        Classifies the dominant color of an object in an image using K-means clustering (kNN-like approach).
        Args:
            image_path (str): Path to the image file.
            colors_json_path (str): Path to the colors.json file.
            n_clusters (int): Number of clusters for K-means (default: 3).
        Returns:
            str: The name of the closest color class.
        """

        if not detections:
            print(f"No objects of desired type detected.")
            return None, 0.0
        # Use the object with the highest confidence (most prominent object)
        best_detection = max(detections, key=lambda x: x['confidence'])

        # Create correct paths for intermediate files
        img_dir = os.path.dirname(image_path)
        img_name = os.path.basename(image_path)

        # Create a subdirectory for processed images to avoid clutter
        processed_dir = os.path.join(img_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        bg_removed_path = os.path.join(processed_dir, f"bg_removed_{img_name}")
        cropped_path = os.path.join(processed_dir, f"cropped_{img_name}")

        self.object_segmenter.remove_background(image_path=image_path, segments=[best_detection], output_path=bg_removed_path)
        crop_image(input_path=bg_removed_path, box=best_detection['box'], output_path=cropped_path)

        # Load image and convert to RGB
        img = Image.open(cropped_path).convert("RGB")
        img_np = np.array(img)
        pixels = img_np.reshape(-1, 3)
        
        #n_clusters = len(self.color_data["colors"])
        n_clusters = 3

        color_classes = [
            {"name": c["name"], "rgb": tuple(int(c["hex"][i:i+2], 16) for i in (1, 3, 5))}
            for c in self.color_data["colors"]
        ]

        # K-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        cluster_centers = kmeans.cluster_centers_
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_idx = labels[np.argmax(counts)]
        dominant_rgb = tuple(map(int, cluster_centers[dominant_idx]))

        # Find closest color class by Euclidean distance
        def color_distance(c1, c2):
            return np.linalg.norm(np.array(c1) - np.array(c2))

        closest = min(color_classes, key=lambda c: color_distance(dominant_rgb, c["rgb"]))
        return closest["name"], None, dominant_rgb

# Example usage:
# color = classify_object_color_knn("object_image.png", "c:/Users/burak/Desktop/DMCAF/assets/colors.json")
# print("Predicted color:", color)
# Example usage:
if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save("example_image.png")
    classifier = ObjectColorClassifier()
    color, confidence = classifier.classify_color("example_image.png", "cat")
    print(f"Predicted color: {color} with confidence: {confidence:.2f}")

