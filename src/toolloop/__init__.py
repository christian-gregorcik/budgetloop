"""The tool-calling loop, written out."""

from .budget import Budget, Stop
from .errors import ArgumentError, SchemaError, ToolError, ToolloopError
from .loop import run
from .messages import Completion, Message, ToolCall
from .schema import build as build_schema
from .tools import Tool, Toolbox, as_tool, tool
from .trace import Run, Step, ToolRun

__all__ = [
    "ArgumentError",
    "Budget",
    "Completion",
    "Message",
    "Run",
    "SchemaError",
    "Step",
    "Stop",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolRun",
    "Toolbox",
    "ToolloopError",
    "as_tool",
    "build_schema",
    "run",
    "tool",
]
__version__ = "0.1.0"
