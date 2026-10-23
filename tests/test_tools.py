from typing import Literal

import pytest

from toolloop import Toolbox, as_tool, tool
from toolloop.errors import ArgumentError, ToolError


@tool
def add(a: int, b: int = 2) -> int:
    """Add two numbers."""
    return a + b


@tool(name="weather", description="Look up the weather.")
def _weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    return f"{city}:{unit}"


def plain(text: str) -> str:
    """Echo the text."""
    return text


def test_the_decorator_keeps_the_function_callable():
    assert add(1, 2) == 3


def test_a_decorated_function_carries_its_tool():
    assert as_tool(add).name == "add"


def test_name_and_description_can_be_set_on_the_decorator():
    entry = as_tool(_weather)
    assert entry.name == "weather"
    assert entry.description == "Look up the weather."


def test_an_undecorated_function_gets_a_schema_on_the_spot():
    assert as_tool(plain).name == "plain"


def test_a_toolbox_keeps_the_order_it_was_given():
    box = Toolbox.of(add, _weather, plain)
    assert [schema["name"] for schema in box.schemas()] == ["add", "weather", "plain"]
    assert len(box) == 3
    assert "add" in box


def test_two_tools_with_one_name_are_refused():
    box = Toolbox.of(add)
    with pytest.raises(ToolError, match="two tools"):
        box.add(add)


def test_an_unknown_tool_lists_the_known_ones():
    box = Toolbox.of(add)
    with pytest.raises(ArgumentError, match="available: add"):
        box.get("subtract")


def test_defaults_are_left_to_the_function():
    assert Toolbox.of(add).get("add").call({"a": 5}) == 7


def test_a_missing_required_argument_is_reported():
    with pytest.raises(ArgumentError, match="'a' is required"):
        Toolbox.of(add).get("add").call({"b": 1})


def test_an_unexpected_argument_is_reported():
    with pytest.raises(ArgumentError, match="unexpected argument 'c'"):
        Toolbox.of(add).get("add").call({"a": 1, "c": 2})


def test_a_number_in_a_string_is_accepted_once():
    assert Toolbox.of(add).get("add").call({"a": "5", "b": "1"}) == 6


def test_a_string_that_is_not_a_number_is_refused():
    with pytest.raises(ArgumentError, match="expected integer"):
        Toolbox.of(add).get("add").call({"a": "five"})


def test_a_float_string_is_not_silently_truncated_into_an_integer():
    with pytest.raises(ArgumentError):
        Toolbox.of(add).get("add").call({"a": "5.5"})


def test_a_bool_is_not_an_integer():
    with pytest.raises(ArgumentError, match="expected integer"):
        Toolbox.of(add).get("add").call({"a": True})


def test_a_value_outside_an_enum_is_refused():
    with pytest.raises(ArgumentError, match="not one of"):
        Toolbox.of(_weather).get("weather").call({"city": "Oslo", "unit": "k"})


def test_null_is_refused_where_it_is_not_allowed():
    with pytest.raises(ArgumentError, match="cannot be null"):
        Toolbox.of(add).get("add").call({"a": None})


def test_null_is_accepted_where_it_is_allowed():
    @tool
    def note(text: str | None) -> str:
        """Write a note."""
        return "empty" if text is None else text

    assert Toolbox.of(note).get("note").call({"text": None}) == "empty"


def test_items_in_a_list_are_checked():
    @tool
    def total(values: list[int]) -> int:
        """Add up a list."""
        return sum(values)

    box = Toolbox.of(total)
    assert box.get("total").call({"values": [1, "2", 3]}) == 6
    with pytest.raises(ArgumentError, match=r"values\[1\]"):
        box.get("total").call({"values": [1, "x"]})


def test_values_in_a_dict_are_checked():
    @tool
    def weigh(scores: dict[str, float]) -> float:
        """Add up the values."""
        return sum(scores.values())

    box = Toolbox.of(weigh)
    assert box.get("weigh").call({"scores": {"a": 1, "b": "2.5"}}) == 3.5
    with pytest.raises(ArgumentError, match=r"scores.b"):
        box.get("weigh").call({"scores": {"a": 1, "b": "x"}})


def test_arguments_must_be_an_object():
    with pytest.raises(ArgumentError, match="must be an object"):
        Toolbox.of(add).get("add").call(["a"])
