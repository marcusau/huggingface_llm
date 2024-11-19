from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from litserve.specs.openai import ChatMessage
import litserve as ls
import base64, torch
from typing import List
from io import BytesIO
from PIL import Image

def decode_base64_image(base64_image_str):
    # Strip the prefix (e.g., 'data:image/jpeg;base64,')
    base64_data = base64_image_str.split(",")[1]
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    return image


class Llama3:
    def __init__(self, device):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16,device_map="auto",)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device

    def apply_chat_template(self, messages: List[ChatMessage]):
        final_messages = []
        image = None
        for message in messages:
            msg = {}
            if message.role == "system":
                msg["role"] = "system"
                msg["content"] = message.content
            elif message.role == "user":
                msg["role"] = "user"
                content = message.content
                final_content = []
                if isinstance(content, list):
                    for i, content in enumerate(content):
                        if content.type == "text":
                            final_content.append(content.dict())
                        elif content.type == "image_url":
                            url = content.image_url.url
                            image = decode_base64_image(url)
                            final_content.append({"type": "image"})
                    msg["content"] = final_content
                else:
                    msg["content"] = content
            elif message.role == "assistant":
                content = message.content
                msg["role"] = "assistant"
                msg["content"] = content
            final_messages.append(msg)
        prompt = self.processor.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, image

    def __call__(self, inputs):
        prompt, image = inputs
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)
        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.2,
            "do_sample": False,
        }

        generate_ids = self.model.generate(
            **inputs,
            **generation_args,
        )
        return inputs, generate_ids

    def decode_tokens(self, outputs):
        inputs, generate_ids = outputs
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

class Llama3VisionAPI(ls.LitAPI):
    def setup(self, device):
        self.model = Llama3(device)

    def decode_request(self, request):
        return self.model.apply_chat_template(request.messages)

    def predict(self, inputs, context):
        yield self.model(inputs)

    def encode_response(self, outputs):
        for output in outputs:
            yield {"role": "assistant", "content": self.model.decode_tokens(output)}

if __name__ == "__main__":
    api = Llama3VisionAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)