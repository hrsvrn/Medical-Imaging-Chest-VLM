from abc import ABC, abstractmethod

class BaseVLM(ABC):
    @abstractmethod
    def load_model(self):
        """Load model and processor/tokenizer"""
        pass

    @abstractmethod
    def infer(self, image, text_prompt=None):
        """Run inference on an image with an optional text prompt"""
        pass
