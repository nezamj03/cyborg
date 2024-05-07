class AttributeDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = AttributeDict(value)  # Convert nested dictionaries
            self.__dict__[key] = value

    def __getattr__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __repr__(self):
        return f"{self.__dict__}"
