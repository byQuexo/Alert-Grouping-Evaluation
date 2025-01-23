import os


def env_or_else(name, else_val=None):
    return os.getenv(name, else_val)


def i_env_or_else(name, else_val=None):
    val = env_or_else(name, else_val)
    if val != "undefined" and val != else_val:
        try:
            return int(val)
        except ValueError:
            return else_val
    else:
        return else_val


config = {
    'http': {
        'port': i_env_or_else("HTTP_PORT", 8007)
    },
    'models': {
        'opus': 'Helsinki-NLP/opus-mt-de-en'
    }
}


def main():
    pass

if __name__ == '__main__':
    main()
