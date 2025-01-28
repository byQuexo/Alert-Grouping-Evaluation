from src.models.base_model import BaseModel


class CDESmallV2(BaseModel):
    def __init__(self, service, config):
        super().__init__(service, config)
        self.name = 'jxm/cde-small-v2'
        self.dim = 768