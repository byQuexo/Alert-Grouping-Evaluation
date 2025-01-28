from src.models.base_model import BaseModel

class GteMultilingualBase(BaseModel):

    def __init__(self, service, config):
        super().__init__(service, config)
        self.name = "Alibaba-NLP/gte-multilingual-base"
        self.dim = 768


