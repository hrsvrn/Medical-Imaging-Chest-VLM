from models.qwen_2_5_vl import QwenVL
from utils.config_loader import load_config
from utils.logging_utils import setup_logger
from PIL import Image

class Inferencer:
    def __init__(self, config_path="configs/qwen2_5_vl.yaml"):
        self.logger = setup_logger()
        cfg = load_config(config_path)
        self.model = QwenVL(
            model_name=cfg["model_name"],
            device=cfg["device"],
            precision=cfg["precision"]
        )
        self.cfg = cfg

        self.logger.info(f"Initializing model: {cfg['model_name']}")
        self.model.load_model()

    def predict(self, image_path, prompt=None):
        image = Image.open(image_path).convert("RGB")
        output = self.model.infer(
            image=image,
            text_prompt=prompt,
            max_new_tokens=self.cfg["max_new_tokens"],
            temperature=self.cfg["temperature"]
        )
        self.logger.info(f"Prompt: {prompt}")
        self.logger.info(f"Output: {output}")
        return output
