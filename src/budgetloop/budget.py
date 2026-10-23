"""What a run is allowed to spend before it stops.

A loop without a ceiling is a loop that bills you for a model's bad afternoon. Every
limit is optional; what is set is enforced, and the run stops with a reason rather than
an exception.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Stop(str, Enum):
    """Why a run ended."""

    ANSWERED = "answered"
    MAX_STEPS = "max_steps"
    MAX_SECONDS = "max_seconds"
    MAX_COST = "max_cost_usd"
    MAX_TOOL_CALLS = "max_tool_calls"

    def __str__(self) -> str:
        return self.value


@dataclass
class Budget:
    """Limits for one run, and the running total against them."""

    max_steps: int = 10
    max_seconds: float | None = None
    max_cost_usd: float | None = None
    max_tool_calls: int | None = None

    steps: int = field(default=0, init=False)
    tool_calls: int = field(default=0, init=False)
    cost_usd: float = field(default=0.0, init=False)
    started: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        for name in ("max_seconds", "max_cost_usd"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.max_tool_calls is not None and self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1 when set")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def spend(self, cost_usd: float | None, tool_calls: int = 0) -> None:
        self.steps += 1
        self.tool_calls += tool_calls
        if cost_usd is not None:
            self.cost_usd += cost_usd

    def exhausted(self) -> Stop | None:
        """The first limit that has been reached, or None while there is room."""
        if self.steps >= self.max_steps:
            return Stop.MAX_STEPS
        if self.max_seconds is not None and self.elapsed >= self.max_seconds:
            return Stop.MAX_SECONDS
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            return Stop.MAX_COST
        if self.max_tool_calls is not None and self.tool_calls >= self.max_tool_calls:
            return Stop.MAX_TOOL_CALLS
        return None
