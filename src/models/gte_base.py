from src.models.base_model import BaseModel


class GteBase(BaseModel):

    def __init__(self, service, config):
        super().__init__(service, config)
        self.name = "Alibaba-NLP/gte-base-en-v1.5"
        self.model_path = f"{self.config['MODELS_PATH']}/{self.name}"
