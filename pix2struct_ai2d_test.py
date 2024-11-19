import os
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image

parent_dir = os.path.dirname(os.path.abspath(__file__))
print(parent_dir)

model_dir = os.path.join(parent_dir, "models/pix2struct_ai2d_base")
processor = Pix2StructProcessor.from_pretrained(pretrained_model_name_or_path=model_dir,
                                                local_files_only=True,)
model = Pix2StructForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_dir,
                                                           local_files_only=True,
                                                           use_safetensors=True)
# Step 1: Load your image
image_path = '/home/marcus/Desktop/project/OCR_transformer_practices/data/testing_1/images/BM00004606/Covering_Letter/page_1_800dpi.png'  # Replace with your actual image path
image = Image.open(image_path)

question = "Extract text from the image"

inputs = processor(images=image, text=question, return_tensors="pt")

predictions = model.generate(**inputs,max_length=1024)

for i in range(len(predictions)):
    print(processor.decode(predictions[i], skip_special_tokens=True))