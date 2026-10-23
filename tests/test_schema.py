from typing import Any, Literal, Optional, Union

import pytest

from toolloop import build_schema
from toolloop.errors import SchemaError
from toolloop.schema import describe


def params(function):
    return build_schema(function)["parameters"]


def test_primitives_and_required():
    def search(query: str, limit: int = 10, ratio: float = 0.5, exact: bool = False) -> str:
        """Search the index."""
        return ""

    schema = params(search)
    assert schema["properties"]["query"] == {"type": "string"}
    assert schema["properties"]["limit"] == {"type": "integer", "default": 10}
    assert schema["properties"]["ratio"]["type"] == "number"
    assert schema["properties"]["exact"]["type"] == "boolean"
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False


def test_the_summary_becomes_the_description():
    def ping() -> str:
        """Check that the service answers.

        It does nothing else.
        """
        return "pong"

    assert build_schema(ping)["description"].startswith("Check that the service answers.")
    assert "nothing else" in build_schema(ping)["description"]


def test_arg_lines_become_parameter_descriptions():
    def send(to: str, body: str) -> None:
        """Send a message.

        Args:
            to: the address
            body: what to say,
                over two lines
        """

    schema = params(send)
    assert schema["properties"]["to"]["description"] == "the address"
    assert schema["properties"]["body"]["description"] == "what to say, over two lines"


def test_literal_becomes_an_enum():
    def pick(mode: Literal["fast", "slow"]) -> str:
        """Pick a mode."""
        return mode

    assert params(pick)["properties"]["mode"] == {"enum": ["fast", "slow"], "type": "string"}


def test_list_and_dict():
    def store(names: list[str], scores: dict[str, float]) -> None:
        """Store things."""

    schema = params(store)
    assert schema["properties"]["names"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["scores"]["additionalProperties"] == {"type": "number"}


def test_optional_is_nullable_and_still_required_without_a_default():
    def note(text: Optional[str]) -> None:  # noqa: UP045
        """Write a note."""

    schema = params(note)
    assert schema["properties"]["text"]["type"] == ["string", "null"]
    assert schema["required"] == ["text"]


def test_new_style_union_with_none():
    def note(text: str | None = None) -> None:
        """Write a note."""

    schema = params(note)
    assert schema["properties"]["text"]["type"] == ["string", "null"]
    assert schema["required"] == []


def test_any_has_no_type():
    def passthrough(value: Any) -> Any:
        """Return what it is given."""
        return value

    assert params(passthrough)["properties"]["value"] == {}


def test_a_missing_hint_is_refused():
    def broken(value) -> None:  # noqa: ANN001
        """No hint."""

    with pytest.raises(SchemaError, match="type hint"):
        build_schema(broken)


def test_varargs_are_refused():
    def broken(*args: int) -> None:
        """Too open."""

    with pytest.raises(SchemaError, match=r"\*args"):
        build_schema(broken)


def test_an_unsupported_type_is_refused():
    def broken(when: complex) -> None:
        """Not describable."""

    with pytest.raises(SchemaError, match="no schema"):
        build_schema(broken)


def test_an_unsupported_union_is_refused():
    def broken(value: Union[int, complex]) -> None:  # noqa: UP007
        """Not describable."""

    with pytest.raises(SchemaError):
        build_schema(broken)


def test_the_name_can_be_overridden():
    def internal_name(a: int) -> int:
        """Doc."""
        return a

    schema = build_schema(internal_name, name="public", description="Another description")
    assert schema["name"] == "public"
    assert schema["description"] == "Another description"


def test_describe_handles_a_docstring_without_args():
    summary, args = describe(lambda: None)
    assert summary == ""
    assert args == {}
