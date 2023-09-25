import os
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, UnidentifiedImageError


def create_images_from_pdf(pdf_path):
    pitchdeck_pdf = pdf_path.split("/")[-1]
    pitchdeck_name = "".join(pitchdeck_pdf.split(".")[:-1])
    images_folder = f"files/images/{pitchdeck_name}"

    images = convert_from_path(pdf_path)
    for idx, image in enumerate(images):
        key = f"{pitchdeck_name}_{idx}.ppm"

        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        file_path = f"{images_folder}/{key}"
        with open(file_path, "wb") as file_object:
            image.save(file_object, "PPM")

    return images_folder


def extract_text_from_image(folder_path: str) -> None:
    file_name = folder_path.split("/")[-1]
    text_file = f"files/ocr/{file_name}.txt"
    if os.path.exists(text_file):
        os.remove(text_file)

    for filename in os.listdir(folder_path):
        if filename.endswith(".ppm"):
            try:
                with open(text_file, "a") as file:
                    image = Image.open(os.path.join(folder_path, filename))
                    text = pytesseract.image_to_string(image)
                    file.write(text)
            except UnidentifiedImageError as e:
                print(f"Error: {e}")
    return text_file
