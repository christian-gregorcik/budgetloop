"""Registering tools, and calling them with arguments that came from a model.

Arguments from a model are untrusted input in the ordinary sense: they are often
almost right. They are checked against the schema and, where a value is obviously the
right thing in the wrong clothing, coerced once. Anything left over is reported back to
the model as a result it can read, not raised at the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .errors import ArgumentError, ToolError
from .schema import build


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    function: Callable[..., Any]

    def call(self, arguments: dict[str, Any]) -> Any:
        """Check the arguments against the schema, then run the function."""
        return self.function(**_prepare(self, arguments))


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Mark a function as a tool. Works bare or with arguments."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        schema = build(fn, name=name, description=description)
        fn.__budgetloop__ = Tool(  # type: ignore[attr-defined]
            name=schema["name"],
            description=schema["description"],
            schema=schema,
            function=fn,
        )
        return fn

    if function is None:
        return wrap
    return wrap(function)


def as_tool(function: Callable[..., Any]) -> Tool:
    """Get the Tool behind a decorated function, or build one on the spot."""
    existing = getattr(function, "__budgetloop__", None)
    if isinstance(existing, Tool):
        return existing
    schema = build(function)
    return Tool(
        name=schema["name"],
        description=schema["description"],
        schema=schema,
        function=function,
    )


@dataclass
class Toolbox:
    """The set of tools a run is allowed to use."""

    tools: dict[str, Tool] = field(default_factory=dict)

    @staticmethod
    def of(*functions: Callable[..., Any]) -> Toolbox:
        box = Toolbox()
        for function in functions:
            box.add(function)
        return box

    def add(self, function: Callable[..., Any]) -> Tool:
        entry = as_tool(function)
        if entry.name in self.tools:
            raise ToolError(f"two tools are called {entry.name!r}")
        self.tools[entry.name] = entry
        return entry

    def __contains__(self, name: object) -> bool:
        return name in self.tools

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.tools.values())

    def get(self, name: str) -> Tool:
        try:
            return self.tools[name]
        except KeyError:
            known = ", ".join(sorted(self.tools)) or "none"
            raise ArgumentError(f"there is no tool called {name!r}; available: {known}") from None

    def schemas(self) -> list[dict[str, Any]]:
        """The schemas to hand a model, in the order the tools were added."""
        return [entry.schema for entry in self.tools.values()]


def _prepare(entry: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ArgumentError(f"{entry.name}: arguments must be an object")

    parameters = entry.schema["parameters"]
    properties: dict[str, Any] = parameters["properties"]
    required: list[str] = parameters["required"]

    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        allowed = ", ".join(properties) or "none"
        raise ArgumentError(f"{entry.name}: unexpected argument {unknown[0]!r}; expected {allowed}")

    missing = [name for name in required if name not in arguments]
    if missing:
        raise ArgumentError(f"{entry.name}: {missing[0]!r} is required")

    return {
        name: _coerce(value, properties[name], f"{entry.name}.{name}")
        for name, value in arguments.items()
    }


_PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _coerce(value: Any, schema: dict[str, Any], where: str) -> Any:
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(option) for option in schema["enum"])
        raise ArgumentError(f"{where}: {value!r} is not one of {allowed}")

    kinds = schema.get("type")
    if kinds is None:
        return value
    if isinstance(kinds, str):
        kinds = [kinds]
    if value is None:
        if "null" in kinds:
            return None
        raise ArgumentError(f"{where}: cannot be null")

    for kind in kinds:
        expected = _PYTHON_TYPES.get(kind)
        if expected is None:
            continue
        if kind != "boolean" and isinstance(value, bool):
            continue
        if isinstance(value, expected):
            return _descend(value, schema, kind, where)

    # A number written as a string is the one mistake worth undoing, because models
    # make it constantly and the intent is never in doubt.
    if isinstance(value, str) and ("integer" in kinds or "number" in kinds):
        try:
            number = float(value)
        except ValueError:
            pass
        else:
            if "integer" in kinds:
                if number.is_integer():
                    return int(number)
            else:
                return number

    wanted = " or ".join(kinds)
    raise ArgumentError(f"{where}: expected {wanted}, got {type(value).__name__}")


def _descend(value: Any, schema: dict[str, Any], kind: str, where: str) -> Any:
    if kind == "array":
        items = schema.get("items")
        if not items:
            return list(value)
        return [_coerce(item, items, f"{where}[{n}]") for n, item in enumerate(value)]
    if kind == "object":
        values = schema.get("additionalProperties")
        if not isinstance(values, dict) or not values:
            return dict(value)
        return {key: _coerce(item, values, f"{where}.{key}") for key, item in value.items()}
    return value
