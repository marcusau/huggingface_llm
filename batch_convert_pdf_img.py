import os
from pathlib import Path
import pymupdf
from PIL import Image
import io
from tqdm import tqdm
# Define paths
base_dir = "/home/marcus/Desktop/project/OCR_transformer_practices/data/testing_1"

pdf_folder = os.path.join(base_dir, "pdf")
images_dir = os.path.join(base_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# Iterate through all subfolders in pdf folder
for pdf_subfolder in tqdm(os.listdir(pdf_folder)):
    pdf_dir = os.path.join(pdf_folder, pdf_subfolder)
    
    # Skip if not a directory
    if not os.path.isdir(pdf_dir):
        continue
        
    # Create corresponding subfolder in images directory
    output_folder = os.path.join(images_dir, pdf_subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all PDFs in current subfolder
    for pdf_file in tqdm(os.listdir(pdf_dir),desc=f'converting {pdf_dir}'):
        if not pdf_file.lower().endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        # Create output subfolder named after the PDF file (without extension)
        pdf_name = Path(pdf_file).stem
        pdf_output_folder = os.path.join(output_folder, pdf_name)
        os.makedirs(pdf_output_folder, exist_ok=True)
        
        with pymupdf.open(pdf_path) as pdf_document:
            # Iterate through each page
            for page_num in tqdm(range(pdf_document.page_count),  desc=f"Converting {pdf_file}"):
                page = pdf_document[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=pymupdf.Matrix(300/72, 300/72))  # 300 DPI
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Save the image
                image_path = os.path.join(pdf_output_folder, f"page_{page_num + 1}.png")
                img.save(image_path, "PNG")


# if __name__ == "__main__":
#     try:
#         image_paths = convert_pdf_to_images(pdf_path)
#         print(f"Successfully converted PDF to images. Images saved in: {os.path.dirname(image_paths[0])}")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
