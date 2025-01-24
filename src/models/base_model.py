
class BaseModel:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


    def create_embeddings(self, text):
        raise NotImplementedError

