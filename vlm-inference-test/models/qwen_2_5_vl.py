import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from .base_vlm import BaseVLM

class QwenVL(BaseVLM):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda", precision="fp16"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.precision = precision
        self.model = None
        self.processor = None

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.precision == "fp16" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

    def infer(self, image, text_prompt="Describe the image.", max_new_tokens=128, temperature=0.7):
        """Run inference using the chat-style input format required by Qwen2.5-VL"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # 1. Build chat-style input
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # 2. Generate chat template (returns text, not tensors)
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # 3. Tokenize both text and image properly
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        # 4. Generate output
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )

        # 5. Decode text
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return output_text[0].strip()
