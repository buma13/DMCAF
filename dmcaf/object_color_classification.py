from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os
import json

class ObjectColorClassifier:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, '..', 'assets', 'colors.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.colors = list(data['colors'])
        
    def classify_color(self, image, object_name):
        color_texts = [f"a photo of a {color} {object_name}" for color in self.colors]
        inputs = self.processor(
            text=color_texts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        best_idx = probs.argmax(dim=1).item()
        return self.colors[best_idx], probs[0, best_idx].item()

def test():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print(probs)

# Example usage:
if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    classifier = ObjectColorClassifier()
    color, confidence = classifier.classify_color(image, "cat")
    print(f"Predicted color: {color} with confidence: {confidence:.2f}")