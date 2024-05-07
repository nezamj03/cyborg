class SimpleLogger:

    def info(self, message: str) -> None:
        print(f"INFO: {message}")

    def stat(self, stat, value, time) -> None:
        print(f"STAT: {stat} @ {time} = {value}")