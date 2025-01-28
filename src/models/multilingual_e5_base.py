from src.models.base_model import BaseModel


class MultilingualE5Base(BaseModel):

    def __init__(self, service, config):
        super().__init__(service, config)
        self.name = "intfloat/multilingual-e5-base"
        self.dim = 768


