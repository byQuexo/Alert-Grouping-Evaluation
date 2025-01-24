import os
from src import Service

os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'src/models/_cache'

def main():
    Service().run()

if __name__ == '__main__':
    main()
