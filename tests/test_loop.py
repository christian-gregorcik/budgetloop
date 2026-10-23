import time

import pytest

from budgetloop import Budget, Completion, Message, Stop, Toolbox, ToolCall, run, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def explode(reason: str) -> str:
    """Always fails."""
    raise RuntimeError(reason)


def script(*completions):
    """A model that returns the given completions in order, then answers."""
    turns = list(completions)

    def complete(messages, tools):
        if turns:
            return turns.pop(0)
        return Completion(content="done")

    return complete


def call(name, arguments, id="1"):
    return ToolCall(id=id, name=name, arguments=arguments)


def test_a_model_that_answers_straight_away():
    result = run(script(Completion(content="hello")), "hi")
    assert result.answer == "hello"
    assert result.stop is Stop.ANSWERED
    assert result.answered
    assert result.tool_calls == 0
    assert len(result.steps) == 1


def test_a_string_prompt_becomes_a_user_message():
    captured = []

    def complete(messages, tools):
        captured.append(list(messages))
        return Completion(content="ok")

    run(complete, "what is the time")
    assert captured[0][0].role == "user"
    assert captured[0][0].content == "what is the time"


def test_a_tool_call_is_run_and_the_result_goes_back():
    model = script(Completion(tool_calls=(call("add", {"a": 2, "b": 3}),)))
    result = run(model, "add two and three", tools=[add])
    assert result.answer == "done"
    assert result.tool_calls == 1
    assert result.steps[0].tool_runs[0].result == "3" or result.steps[0].tool_runs[0].result == "5"
    tool_message = [m for m in result.messages if m.role == "tool"][0]
    assert tool_message.content == "5"
    assert tool_message.tool_call_id == "1"


def test_the_schemas_reach_the_model():
    seen = []

    def complete(messages, tools):
        seen.append(tools)
        return Completion(content="ok")

    run(complete, "hi", tools=[add])
    assert seen[0][0]["name"] == "add"


def test_several_calls_in_one_turn_all_run():
    calls = (call("add", {"a": 1, "b": 1}, "a"), call("add", {"a": 2, "b": 2}, "b"))
    result = run(script(Completion(tool_calls=calls)), "hi", tools=[add])
    assert [r.result for r in result.steps[0].tool_runs] == ["2", "4"]
    assert [m.tool_call_id for m in result.messages if m.role == "tool"] == ["a", "b"]


def test_a_tool_that_raises_is_reported_to_the_model_not_the_caller():
    model = script(Completion(tool_calls=(call("explode", {"reason": "disk full"}),)))
    result = run(model, "hi", tools=[explode])
    assert result.answered
    assert result.failed_tool_calls == 1
    assert "disk full" in result.steps[0].tool_runs[0].result


def test_bad_arguments_come_back_as_a_result_the_model_can_read():
    model = script(Completion(tool_calls=(call("add", {"a": "x", "b": 1}),)))
    result = run(model, "hi", tools=[add])
    run_detail = result.steps[0].tool_runs[0]
    assert run_detail.failed
    assert "expected integer" in run_detail.result


def test_an_unknown_tool_comes_back_as_a_result():
    model = script(Completion(tool_calls=(call("divide", {"a": 1}),)))
    result = run(model, "hi", tools=[add])
    assert "no tool called 'divide'" in result.steps[0].tool_runs[0].result


def test_the_loop_stops_at_max_steps():
    def never_answers(messages, tools):
        return Completion(tool_calls=(call("add", {"a": 1, "b": 1}),))

    result = run(never_answers, "hi", tools=[add], budget=Budget(max_steps=3))
    assert result.stop is Stop.MAX_STEPS
    assert len(result.steps) == 3


def test_the_loop_stops_on_cost():
    def never_answers(messages, tools):
        return Completion(tool_calls=(call("add", {"a": 1, "b": 1}),), cost_usd=0.01)

    result = run(never_answers, "hi", tools=[add], budget=Budget(max_steps=50, max_cost_usd=0.025))
    assert result.stop is Stop.MAX_COST
    assert len(result.steps) == 3
    assert result.cost_usd == pytest.approx(0.03)


def test_the_loop_stops_on_tool_calls():
    def never_answers(messages, tools):
        return Completion(tool_calls=(call("add", {"a": 1, "b": 1}),))

    result = run(never_answers, "hi", tools=[add], budget=Budget(max_steps=50, max_tool_calls=2))
    assert result.stop is Stop.MAX_TOOL_CALLS


def test_the_loop_stops_on_time():
    def slow(messages, tools):
        time.sleep(0.02)
        return Completion(tool_calls=(call("add", {"a": 1, "b": 1}),))

    result = run(slow, "hi", tools=[add], budget=Budget(max_steps=50, max_seconds=0.05))
    assert result.stop is Stop.MAX_SECONDS


def test_a_run_that_hits_a_limit_is_not_answered():
    def never_answers(messages, tools):
        return Completion(tool_calls=(call("add", {"a": 1, "b": 1}),))

    assert not run(never_answers, "hi", tools=[add], budget=Budget(max_steps=1)).answered


def test_history_is_passed_through_and_grows():
    start = [Message.system("be brief"), Message.user("hi")]
    result = run(
        script(Completion(tool_calls=(call("add", {"a": 1, "b": 1}),))), start, tools=[add]
    )
    roles = [message.role for message in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_the_caller_can_pass_a_toolbox():
    box = Toolbox.of(add)
    result = run(script(Completion(tool_calls=(call("add", {"a": 1, "b": 1}),))), "hi", tools=box)
    assert result.tool_calls == 1


def test_a_listener_sees_each_step():
    seen = []
    run(
        script(Completion(tool_calls=(call("add", {"a": 1, "b": 1}),))),
        "hi",
        tools=[add],
        on_step=lambda step: seen.append(step.index),
    )
    assert seen == [0, 1]


def test_a_non_string_result_is_rendered_as_json():
    @tool
    def rows(n: int) -> list[dict]:
        """Return rows."""
        return [{"id": i} for i in range(n)]

    result = run(script(Completion(tool_calls=(call("rows", {"n": 2}),))), "hi", tools=[rows])
    assert result.steps[0].tool_runs[0].result == '[{"id": 0}, {"id": 1}]'


def test_the_renderer_can_be_replaced():
    @tool
    def rows(n: int) -> list[int]:
        """Return rows."""
        return list(range(n))

    model = script(Completion(tool_calls=(call("rows", {"n": 3}),)))
    result = run(model, "hi", tools=[rows], render=lambda value: f"{len(value)} rows")
    assert result.steps[0].tool_runs[0].result == "3 rows"


def test_workers_must_be_at_least_one():
    with pytest.raises(ValueError):
        run(script(Completion(content="hi")), "hi", workers=0)


def test_the_trace_is_plain_data():
    model = script(
        Completion(content="thinking", tool_calls=(call("add", {"a": 1, "b": 1}),), cost_usd=0.002)
    )
    data = run(model, "hi", tools=[add]).to_dict()
    assert data["stop"] == "answered"
    assert data["answer"] == "done"
    assert data["tool_calls"] == 1
    assert data["steps"][0]["tools"][0]["name"] == "add"
    assert data["cost_usd"] == 0.002
    assert data["messages"][1]["tool_calls"][0]["arguments"] == {"a": 1, "b": 1}
