import base64
import io
import os
from PIL import Image
import ollama

# Set the OLLAMA_HOST environment variable
os.environ["OLLAMA_HOST"] = "http://localhost:11435"



def encode_image_to_base64(image_path: str, format: str = "PNG") -> str:
    """Encodes an image file to a base64 string.

    Args:
        image_path (str): Path to the image file.
        format (str): Format to save the image in memory (default is PNG).

    Returns:
        str: Base64-encoded image.
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    

# def get_ocr_output_from_image(image_base64: str, model: str = "mistral:latest") -> str:
#     """Sends an image to the Llama OCR model and returns structured text output.

#     Args:
#         image_base64 (str): Base64-encoded image string.
#         model (str): The model version to use for OCR (default is latest Llama 3.2 Vision).

#     Returns:
#         str: Extracted and structured text from the image.
#     """
#     response = ollama.chat(
#         model=model,
#         messages=[{
#             "role": "user",
#             "content": "extract the text from the image",
#             "images": [image_base64]
#         }]
#     )
#     return response.get('message', {}).get('content', '').strip()

if __name__ == "__main__":
     # Set your desired parameters
     options = {
        "temperature": 0.0,  # Adjust temperature as needed (0.0 to 1.0)
        "max_tokens": 1000,    # Optional: Limit the number of tokens in the response
        "top_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 0
     }
     image_path = '/home/marcus/Desktop/project/OCR_transformer_practices/data/testing_1/images/BM00004606/Covering_Letter/page_1_500dpi.png' 
     base64_image = encode_image_to_base64(image_path)
     response = ollama.chat(
        model="llava:7b",
        messages=[{
            "role": "user",
            "content": "Extract the texts from the picture. Don't describe the image.",
            "images": [base64_image]
        }]
        )
     response_text = response.get('message', {}).get('content', '').strip()
     print(response_text)