from qdrant_client import QdrantClient

class Qdrant:
    def __init__(self):
        self.client = QdrantClient(
            host="localhost",
            port=6333
        )

    