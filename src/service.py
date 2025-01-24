from src.clients.qdrant import Qdrant
from src.config.app import App

class Service:
    def __init__(self):
        self.config = App.config()
        self.qdrant = Qdrant(self, self.config)

    def run(self):
        print(self.config)