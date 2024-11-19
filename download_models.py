from huggingface_hub import snapshot_download

# # Specify the model ID and revision
# model_id = "vikhyatk/moondream2"
# revision = "2024-08-26"

# # Specify the directory where you want to download the model
# download_directory = "/home/marcus/Desktop/project/OCR_transformer_practices/models/moondream2"  # Change this to your desired path

# # Download the model files to the specified directory
# local_model_path = snapshot_download(repo_id=model_id, revision=revision, local_dir=download_directory)


# ########################################################################
# from transformers import AutoModel, AutoTokenizer

# # Define the model ID
# model_id = "microsoft/Phi-3-vision-128k-instruct"

# # Load the model and tokenizer
# model = AutoModel.from_pretrained(model_id, trust_remote_code=True, _attn_implementation="eager")
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# # Save the model and tokenizer locally
# model.save_pretrained("./home/marcus/Desktop/project/OCR_transformer_practices/models/Phi_3_vision_128k_instruct")
# tokenizer.save_pretrained("/home/marcus/Desktop/project/OCR_transformer_practices/models/Phi_3_vision_128k_instruct")

##########################################################################


# from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


# model_id="google/pix2struct-ai2d-base"
# model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
# processor = Pix2StructProcessor.from_pretrained(model_id)

# model.save_pretrained("/home/marcus/Desktop/project/OCR_transformer_practices/models/pix2struct_ai2d_base")
# processor.save_pretrained("/home/marcus/Desktop/project/OCR_transformer_practices/models/pix2struct_ai2d_base")

###################################################################


# import os
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import requests

# # Set up directories
# local_model_dir = '/home/marcus/Desktop/project/OCR_transformer_practices/models/trocr_large_printed'  # Define your local directory

# # Step 1: Download and save the model and processor locally if not already done

# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

#     # Save the model and processor to the specified directory
# processor.save_pretrained(local_model_dir)
# model.save_pretrained(local_model_dir)

#################################################################################3

# from huggingface_hub import snapshot_download

# # Specify the model ID
# model_id = "meta-llama/Llama-3.2-11B-Vision"

# # Download the model to a local directory
# snapshot_download(repo_id=model_id, local_dir="/home/marcus/Desktop/project/OCR_transformer_practices/models/Llama_32_11B_Vision")

####################################################################################

from huggingface_hub import snapshot_download

# Specify the model ID
model_id = "impactframes/Llama-3.2-11B-Vision-bnb-4bit"

# Download the model to a local directory
snapshot_download(repo_id=model_id, local_dir="/home/marcus/Desktop/project/OCR_transformer_practices/models/Llama_32_11B_Vision_bnb_4bit")