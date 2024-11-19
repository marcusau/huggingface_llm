from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import os
import torch

# Get the parent directory
project_dir = Path(__file__).parent
model_folder_name = 'models/moondream2'
model_dir = str(project_dir/model_folder_name)


# Load the tokenizer and model using the correct model ID
# model_id = "vikhyatk/moondream2"
# revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16, ).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_dir ,)

# Now you can uncomment and use the rest of your code for image processing
image = Image.open(os.path.join(project_dir, 'data/testing_1/images/BM00004606/Covering_Letter/page_1_800dpi.png'))

# Encode the image using the model (this step might vary depending on your model's API)
enc_image = model.encode_image(image)

# Provide a prompt asking for text recognition
input_prompt = "Extract all texts from the image."

response = model.answer_question(enc_image,input_prompt, tokenizer)

print(response)