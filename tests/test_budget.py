import time

import pytest

from toolloop import Budget, Stop


def test_a_fresh_budget_has_room():
    assert Budget(max_steps=2).exhausted() is None


def test_steps_are_counted():
    budget = Budget(max_steps=2)
    budget.spend(None)
    assert budget.exhausted() is None
    budget.spend(None)
    assert budget.exhausted() is Stop.MAX_STEPS


def test_cost_is_summed_and_unknown_cost_is_skipped():
    budget = Budget(max_steps=10, max_cost_usd=0.01)
    budget.spend(0.004)
    budget.spend(None)
    assert budget.cost_usd == pytest.approx(0.004)
    budget.spend(0.006)
    assert budget.exhausted() is Stop.MAX_COST


def test_tool_calls_are_counted():
    budget = Budget(max_steps=10, max_tool_calls=3)
    budget.spend(None, tool_calls=2)
    assert budget.exhausted() is None
    budget.spend(None, tool_calls=1)
    assert budget.exhausted() is Stop.MAX_TOOL_CALLS


def test_time_runs_out():
    budget = Budget(max_steps=10, max_seconds=0.01)
    time.sleep(0.02)
    assert budget.exhausted() is Stop.MAX_SECONDS


def test_steps_are_reported_before_the_softer_limits():
    budget = Budget(max_steps=1, max_cost_usd=0.001)
    budget.spend(0.01)
    assert budget.exhausted() is Stop.MAX_STEPS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"max_seconds": 0},
        {"max_cost_usd": -1},
        {"max_tool_calls": 0},
    ],
)
def test_nonsense_limits_are_refused(kwargs):
    with pytest.raises(ValueError):
        Budget(**kwargs)


def test_stop_prints_as_its_value():
    assert str(Stop.ANSWERED) == "answered"
    assert f"{Stop.MAX_COST}" == "max_cost_usd"
