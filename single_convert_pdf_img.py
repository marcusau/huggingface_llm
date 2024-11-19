import os
from pathlib import Path
import pymupdf
from PIL import Image
import io
from tqdm import tqdm
# Define paths
# Get input PDF path from user
pdf_path = "data/testing_1/pdf/BM00004606/Covering Letter.pdf"

# Extract base directory and PDF filename
base_dir = str(Path(pdf_path).parent.parent.parent)
print(f"Base directory: {base_dir}")
pdf_name = Path(pdf_path).stem.replace(" ", "_").replace("-", "_")
print(f"PDF name: {pdf_name}")
pdf_folder = str(Path(pdf_path).parent.parent)
print(f"PDF folder: {pdf_folder}")
pdf_sub_folder = str(Path(pdf_path).parent)
print(f"PDF sub folder: {pdf_sub_folder}")
# # Create images directory parallel to PDF folder
images_dir = os.path.join(base_dir, "images")
print(f"Images directory: {images_dir}")
os.makedirs(images_dir, exist_ok=True)

# # Create subfolder in images directory matching PDF parent folder name
pdf_subfolder_name = str(Path(pdf_path).parent.stem)
print(f"PDF subfolder name: {pdf_subfolder_name}")
output_folder = os.path.join(images_dir, pdf_subfolder_name)
print(f"Output folder: {output_folder}")
os.makedirs(output_folder, exist_ok=True)

# # Create output subfolder named after the PDF file
pdf_output_folder = os.path.join(output_folder, pdf_name)
print(f"PDF output folder: {pdf_output_folder}")
os.makedirs(pdf_output_folder, exist_ok=True)
# Convert PDF pages to images
with pymupdf.open(pdf_path) as pdf_document:
    # Iterate through each page
    for page_num in tqdm(range(pdf_document.page_count), desc=f"Converting {pdf_name}"):
        page = pdf_document[page_num]
        
        # Convert page to image
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1200/72, 1200/72))  # 1200 DPI
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Save the image
        image_path = os.path.join(pdf_output_folder, f"page_{page_num + 1}_1_200dpi.png")
        img.save(image_path, "PNG")
