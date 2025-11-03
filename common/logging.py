
import logging


class CountingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.warnings = 0
        self.errors = 0

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.errors += 1
        elif record.levelno >= logging.WARNING:
            self.warnings += 1