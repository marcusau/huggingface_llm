import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

local_model_dir = '/home/marcus/Desktop/project/OCR_transformer_practices/models/trocr_large_printed'
# Step 2: Load the model and processor from local directory
processor = TrOCRProcessor.from_pretrained(pretrained_model_name_or_path=local_model_dir,local_files_only=True,force_download=False, )
model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path=local_model_dir,local_files_only=True,force_download=False, )
# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load image from a URL (you can also load from a local path)
image_dir = '/home/marcus/Desktop/project/OCR_transformer_practices/data/testing_1/images/BM00004606/Covering_Letter/page_1_500dpi.png'
image = Image.open(image_dir).convert("RGB")

# Step 3: Process the image
pixel_values = processor(images=image, return_tensors="pt").pixel_values
# Move pixel values to GPU
pixel_values = pixel_values.to(device)
# Step 4: Generate text from the image
generated_ids = model.generate(pixel_values,max_new_tokens=1024)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

# Print the generated text
for text in generated_text:
    print(text)