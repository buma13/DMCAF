
from rembg import remove
from PIL import Image

def remove_image_background(input_path: str, output_path: str=None) -> Image.Image:
    """
    Removes the background from an image at input_path and saves the result to output_path.
    """
    input = Image.open(input_path)
    output = remove(input)
    if output_path:
        output.save(output_path)
    return output

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

if __name__ == "__main__":
    remove_image_background(input_path="coco_sample.jpg", output_path="rembg_output.png")