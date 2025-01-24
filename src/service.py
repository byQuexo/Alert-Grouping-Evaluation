from src.models import *
from src.clients import Qdrant
from src.config import App

from loguru import logger

class Service:
    def __init__(self):
        self.config = App.config()
        self.qdrant = Qdrant(self, self.config)
        self.model_manager = ModelManager(self, self.config)

    def run(self):
        try:
            self.init()
        except Exception as e:
            logger.error(f"Error running service: {e}")
            raise e


    def init(self):
        try:
            self.model_manager.setup()
            logger.success("Setups for Models successfully completed.")
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise e



