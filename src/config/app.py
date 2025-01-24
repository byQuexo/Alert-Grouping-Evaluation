class App:
    __conf = {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": 6333,
        "MODELS_PATH": "src/models/_cache",
    }

    @staticmethod
    def config():
        return App.__conf
