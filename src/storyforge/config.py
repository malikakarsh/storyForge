import tomllib

def load_config(path):
    with open(path, "rb") as file:
        config = tomllib.load(file)

    return config