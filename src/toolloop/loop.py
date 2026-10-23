"""The loop itself.

Ask the model. If it wants tools, run them, hand the results back, ask again. Stop when
it answers in words or when the budget runs out.

The model is a callable you supply:

    complete(messages: list[Message], tools: list[dict]) -> Completion

Nothing here knows about a provider, which is why the whole loop can be tested with a
scripted model and no network.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .budget import Budget, Stop
from .errors import ArgumentError
from .messages import Completion, Message, ToolCall
from .tools import Tool, Toolbox
from .trace import Run, Step, ToolRun

Complete = Callable[[list[Message], list[dict[str, Any]]], Completion]
Render = Callable[[Any], str]


def _default_render(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    import json

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _run_one(box: Toolbox, call: ToolCall, render: Render) -> ToolRun:
    started = time.perf_counter()
    try:
        entry: Tool = box.get(call.name)
        result = render(entry.call(call.arguments))
        failed = False
    except ArgumentError as exc:
        result, failed = f"error: {exc}", True
    except Exception as exc:  # a tool that throws is the model's problem to work around
        result, failed = f"error: {type(exc).__name__}: {exc}", True
    return ToolRun(
        call=call, result=result, failed=failed, ms=(time.perf_counter() - started) * 1000
    )


def run(
    complete: Complete,
    messages: Sequence[Message] | str,
    tools: Toolbox | Sequence[Callable[..., Any]] = (),
    budget: Budget | None = None,
    workers: int = 4,
    render: Render = _default_render,
    on_step: Callable[[Step], None] | None = None,
) -> Run:
    """Run the loop until the model answers or the budget is spent."""
    if workers < 1:
        raise ValueError("workers must be at least 1")

    box = tools if isinstance(tools, Toolbox) else Toolbox.of(*tools)
    history = [Message.user(messages)] if isinstance(messages, str) else list(messages)
    limits = budget or Budget()
    schemas = box.schemas()

    started = time.perf_counter()
    steps: list[Step] = []
    stop = Stop.MAX_STEPS

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            turn_started = time.perf_counter()
            completion = complete(history, schemas)
            history.append(Message.assistant(completion.content, completion.tool_calls))

            step = Step(index=len(steps), completion=completion)
            if completion.wants_tools:
                calls = list(completion.tool_calls)
                step.tool_runs = list(pool.map(lambda call: _run_one(box, call, render), calls))
                for tool_run in step.tool_runs:
                    history.append(Message.tool(tool_run.call, tool_run.result))

            step.ms = (time.perf_counter() - turn_started) * 1000
            steps.append(step)
            limits.spend(completion.cost_usd, tool_calls=len(step.tool_runs))
            if on_step is not None:
                on_step(step)

            if not completion.wants_tools:
                stop = Stop.ANSWERED
                break
            reached = limits.exhausted()
            if reached is not None:
                stop = reached
                break

    return Run(
        messages=history,
        steps=steps,
        stop=stop,
        cost_usd=limits.cost_usd,
        ms=(time.perf_counter() - started) * 1000,
    )
