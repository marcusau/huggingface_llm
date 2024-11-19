import base64
import io
import os
from PIL import Image
import ollama

# Set the OLLAMA_HOST environment variable
os.environ["OLLAMA_HOST"] = "http://localhost:11435"

image_path = "/home/marcus/Desktop/project/OCR_transformer_practices/data/testing_1/images/BM00004606/Covering_Letter/page_1_500dpi.png"

with Image.open(image_path) as img:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    base64_image=base64.b64encode(buffered.getvalue()).decode('utf-8')


options = {
        "temperature": 0.0,  # Adjust temperature as needed (0.0 to 1.0)
        "max_tokens": 1000,    # Optional: Limit the number of tokens in the response
        "top_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 0
     }
response = ollama.chat(
        model="llama3.2-vision:latest",
        messages=[{
            "role": "user",
            "content": "Extract and return only the plain text exactly as it appears in the image, without adding any descriptions or interpretations. And please avoidadding any formatting symbols such as asterisks (*), bold (**), or other special characters.",
            "images": [base64_image]
        }]
        )
response_text = response.get('message', {}).get('content', '').strip()
print(response_text)