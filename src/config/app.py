class App:
    __conf = {
        "QDRANT": {
            "HOST": "localhost",
            "PORT": 6333,
        },
        "CACHE_DIR": "src/models/_cache",
        "QDRANT_COLLECTION_DIM": 768,
        "DATA": {
            "LANGUAGES": ["English", "German", "French", "Spanish"],
            "VALIDATION": "validation",
        }
    }

    @staticmethod
    def config():
        return App.__conf
