from typing import Dict
from loguru import logger

from src.models.base_model import BaseModel
from src.models.cde_small_v2 import CDESmallV2
from src.models.gte_base import GteBase
from src.models.gte_multilingual_base import GteMultilingualBase
from src.models.multilingual_e5_base import MultilingualE5Base


class ModelManager:
    def __init__(self, service, config):
        """Initialize the model manager with available models."""
        self.models: Dict[str, BaseModel] = {}
        self.available_models = {
            "Alibaba-NLP/gte-base-en-v1.5": GteBase,
            "Alibaba-NLP/gte-multilingual-base": GteMultilingualBase,
            "intfloat/multilingual-e5-base": MultilingualE5Base,
            "jxm/cde-small-v2": CDESmallV2,
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

