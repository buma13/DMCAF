
from PIL import Image

def crop_image(input_path: str, box: list, output_path: str=None) -> Image.Image:
    """
    Crops the image at input_path using bounding box coordinates [x1, y1, x2, y2]
    and saves the cropped image to output_path.
    """
    image = Image.open(input_path)
    cropped = image.crop((box[0], box[1], box[2], box[3]))
    if output_path:
        cropped.save(output_path)
    return cropped
