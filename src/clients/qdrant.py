from qdrant_client import QdrantClient

class Qdrant:
    def __init__(self, service, config):
        self.service  = service
        self.config = config
        self.client = QdrantClient(
            host=self.config['QDRANT_HOST'],
            port=self.config['QDRANT_PORT'],
        )
    