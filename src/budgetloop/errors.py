"""Errors raised on purpose."""


class ToolloopError(Exception):
    """Base class for every error this package raises."""


class SchemaError(ToolloopError):
    """A tool was written in a way that no schema can be built from."""


class ToolError(ToolloopError):
    """A tool was registered or looked up wrongly."""


class ArgumentError(ToolloopError):
    """The model called a tool with arguments that do not fit it.

    This one is caught inside the loop and handed back to the model as a result, so it
    can correct itself. It is not a crash.
    """
