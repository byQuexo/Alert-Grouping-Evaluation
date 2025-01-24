class App:
    __conf = {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": 6333,
    }

    @staticmethod
    def config():
        return App.__conf
