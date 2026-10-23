"""Turning a Python function into the JSON Schema a model is given.

The signature is the source of truth. Writing the schema by hand next to the function
is how the two drift apart, and a drifted schema shows up as a model that calls the
tool wrongly for reasons nobody can see.
"""

from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin  # noqa: UP035

from .errors import SchemaError

_PRIMITIVES: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}

_NO_DEFAULT = inspect.Parameter.empty


@dataclass(frozen=True)
class Parameter:
    name: str
    schema: dict[str, Any]
    required: bool


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return the annotation without ``None``, and whether ``None`` was allowed."""
    origin = get_origin(annotation)
    if origin is not Union and origin is not types.UnionType:
        return annotation, False
    members = [member for member in get_args(annotation) if member is not type(None)]
    if len(members) != len(get_args(annotation)):
        if len(members) == 1:
            return members[0], True
        rebuilt = members[0]
        for member in members[1:]:
            rebuilt = rebuilt | member
        return rebuilt, True
    return annotation, False


def type_schema(annotation: Any, where: str) -> dict[str, Any]:
    """Build the schema for one annotation."""
    annotation, nullable = _unwrap_optional(annotation)

    schema = _bare_type_schema(annotation, where)
    if nullable and "type" in schema and isinstance(schema["type"], str):
        schema["type"] = [schema["type"], "null"]
    return schema


def _bare_type_schema(annotation: Any, where: str) -> dict[str, Any]:
    if annotation is Any or annotation is inspect.Parameter.empty:
        return {}
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)
    if origin is Literal:
        values = list(get_args(annotation))
        kinds = {_PRIMITIVES.get(type(value)) for value in values}
        schema: dict[str, Any] = {"enum": values}
        if len(kinds) == 1 and None not in kinds:
            schema["type"] = kinds.pop()
        return schema
    if origin in (list, set, tuple):
        (item,) = get_args(annotation) or (Any,)
        return {"type": "array", "items": type_schema(item, where)}
    if origin is dict:
        args = get_args(annotation)
        value = args[1] if len(args) == 2 else Any
        return {"type": "object", "additionalProperties": type_schema(value, where)}

    raise SchemaError(
        f"{where}: no schema for {annotation!r}; use str, int, float, bool, "
        "list, dict, Literal, or a union with None"
    )


def describe(function: Any) -> tuple[str, dict[str, str]]:
    """Split a docstring into its summary and its per-parameter lines.

    Understands the ``Args:`` block used by Google-style docstrings. Anything it does
    not understand is left in the summary, which is the safe way round.
    """
    doc = inspect.getdoc(function) or ""
    lines = doc.splitlines()
    summary: list[str] = []
    params: dict[str, str] = {}
    current: str | None = None
    in_args = False

    for line in lines:
        stripped = line.strip()
        if not in_args and stripped.lower() in {"args:", "arguments:", "parameters:"}:
            in_args = True
            continue
        if in_args:
            if stripped and not line.startswith((" ", "\t")):
                in_args = False
                summary.append(line)
                continue
            if ":" in stripped and not stripped.startswith(("-", "*")):
                name, _, text = stripped.partition(":")
                current = name.strip().split(" ")[0]
                params[current] = text.strip()
            elif current and stripped:
                params[current] = f"{params[current]} {stripped}".strip()
            continue
        summary.append(line)

    return "\n".join(summary).strip(), params


def build(function: Any, name: str | None = None, description: str | None = None) -> dict[str, Any]:
    """Build the whole tool schema for a function."""
    tool_name = name or function.__name__
    signature = inspect.signature(function)
    try:
        hints = typing.get_type_hints(function)
    except Exception as exc:
        raise SchemaError(f"{tool_name}: cannot read the type hints: {exc}") from None

    summary, param_docs = describe(function)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter in signature.parameters.values():
        where = f"{tool_name}.{parameter.name}"
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise SchemaError(f"{where}: *args and **kwargs cannot be described to a model")
        if parameter.name not in hints:
            raise SchemaError(f"{where}: every parameter needs a type hint")

        schema = type_schema(hints[parameter.name], where)
        if parameter.name in param_docs:
            schema["description"] = param_docs[parameter.name]
        if parameter.default is not _NO_DEFAULT:
            schema["default"] = parameter.default
        else:
            required.append(parameter.name)
        properties[parameter.name] = schema

    return {
        "name": tool_name,
        "description": description or summary,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }
