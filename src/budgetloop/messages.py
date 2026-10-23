"""The small set of shapes that travel between the loop and a model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One entry in the conversation.

    An assistant turn carries ``tool_calls`` when the model wants tools run. A tool turn
    carries ``tool_call_id`` and the result as text.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @staticmethod
    def system(content: str) -> Message:
        return Message(role="system", content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message(role="user", content=content)

    @staticmethod
    def assistant(content: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return Message(role="assistant", content=content, tool_calls=tool_calls)

    @staticmethod
    def tool(call: ToolCall, content: str) -> Message:
        return Message(role="tool", content=content, tool_call_id=call.id, name=call.name)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        return data


@dataclass(frozen=True)
class Completion:
    """What a model hands back for one turn.

    ``cost_usd`` and the token counts are optional. The budget only enforces what it is
    told about: a model that reports nothing can still be stopped by steps or by time.
    """

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    cost_usd: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw: Any = field(default=None, repr=False, compare=False)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)
