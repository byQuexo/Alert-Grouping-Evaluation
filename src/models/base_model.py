import os
from loguru import logger
from sentence_transformers import SentenceTransformer
from abc import ABC
import torch


class BaseModel(ABC):
    """Abstract base class for all embedding models."""

    def __init__(self, service, config):
        self.service = service
        self.config = config
        self.name = None
        self.model_path = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup(self):
        try:
            if not os.path.exists(self.model_path):
                self.model = SentenceTransformer(self.name, trust_remote_code=True, device=self.device)
                self.model.save_pretrained(self.model_path)
                logger.info(f"Download for Model {self.name} completed.")
            else:
                self.model = SentenceTransformer(self.model_path, trust_remote_code=True, device=self.device)
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise e


    def create_embedding(self, text) -> list[float]:
        try:
            logger.info(f"Creating embedding for text: {text[:30]}")
            return self.model.encode(text, convert_to_tensor=True, precision="float32").tolist()
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise e
