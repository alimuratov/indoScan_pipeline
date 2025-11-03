class StepPreconditionError(Exception):
    def __init__(self, code: str, message: str, context: str = ""):
        super().__init__(message)
        self.code = code
        self.context = context