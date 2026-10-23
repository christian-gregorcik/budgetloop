"""A record of what happened, in the order it happened."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .budget import Stop
from .messages import Completion, Message, ToolCall


@dataclass(frozen=True)
class ToolRun:
    """One tool call and what came back from it."""

    call: ToolCall
    result: str
    failed: bool
    ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.call.id,
            "name": self.call.name,
            "arguments": self.call.arguments,
            "result": self.result,
            "failed": self.failed,
            "ms": round(self.ms, 1),
        }


@dataclass
class Step:
    """One turn: what the model said, and what its tools did."""

    index: int
    completion: Completion
    tool_runs: list[ToolRun] = field(default_factory=list)
    ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.index,
            "content": self.completion.content,
            "cost_usd": self.completion.cost_usd,
            "ms": round(self.ms, 1),
            "tools": [run.to_dict() for run in self.tool_runs],
        }


@dataclass
class Run:
    """The finished run: the answer, why it ended, and everything it took."""

    messages: list[Message]
    steps: list[Step]
    stop: Stop
    cost_usd: float
    ms: float

    @property
    def answer(self) -> str:
        for message in reversed(self.messages):
            if message.role == "assistant" and message.content:
                return message.content
        return ""

    @property
    def answered(self) -> bool:
        return self.stop is Stop.ANSWERED

    @property
    def tool_calls(self) -> int:
        return sum(len(step.tool_runs) for step in self.steps)

    @property
    def failed_tool_calls(self) -> int:
        return sum(1 for step in self.steps for run in step.tool_runs if run.failed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop": str(self.stop),
            "answer": self.answer,
            "steps": [step.to_dict() for step in self.steps],
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "cost_usd": round(self.cost_usd, 6),
            "ms": round(self.ms, 1),
            "messages": [message.to_dict() for message in self.messages],
        }
