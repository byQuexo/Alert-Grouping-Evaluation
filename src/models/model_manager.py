from src.models import GteBase
from src.models import GteMultilingualBase
from src.models import BaseModel

from loguru import logger


class ModelManager:
    def __init__(self, service, config):
        """Initialize the model manager with available models."""
        self.models: Dict[str, BaseModel] = {}
        self.available_models = {
            "Alibaba-NLP/gte-base-en-v1.5": GteBase,
            "Alibaba-NLP/gte-multilingual-base": GteMultilingualBase,
        }

        self.service = service
        self.config = config


    def setup(self):
        try:
            for name, model in self.available_models.items():
                model_instance = model(self.service, self.config)
                model_instance.setup()
                self.models[name] = model_instance
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise e

    def get_model(self, name: str) -> BaseModel:
        return self.models[name]

    def create_embedding(self, model_name: str, text: str) -> list[float]:
        """Create an embedding for the text."""
        try:
            model = self.get_model(model_name)
            return model.create_embedding(text)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise e

