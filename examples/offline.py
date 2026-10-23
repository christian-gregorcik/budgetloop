"""The loop end to end, with a scripted model so it runs offline.

    python examples/offline.py

The model here asks for two tools at once, gets one good result and one error, and
recovers. That is the behaviour worth seeing: a tool that fails is a message the model
can read, not a crash.
"""

from __future__ import annotations

import json

from toolloop import Budget, Completion, ToolCall, run, tool

PRICES = {"SE0000108656": 118.4, "SE0000148884": 236.0}


@tool
def price(isin: str) -> float:
    """Look up the last price for an instrument.

    Args:
        isin: the twelve-character identifier
    """
    try:
        return PRICES[isin]
    except KeyError:
        raise LookupError(f"{isin} is not in the price list") from None


@tool
def convert(amount: float, rate: float) -> float:
    """Convert an amount at a given rate.

    Args:
        amount: the amount to convert
        rate: how many units of the target currency one unit buys
    """
    return round(amount * rate, 2)


def scripted_model(messages, tools):
    """Stands in for a provider. Decides what to do from how far along it is."""
    step = sum(1 for message in messages if message.role == "assistant")

    if step == 0:
        return Completion(
            content="Looking both of them up.",
            tool_calls=(
                ToolCall(id="a", name="price", arguments={"isin": "SE0000108656"}),
                ToolCall(id="b", name="price", arguments={"isin": "SE0000000000"}),
            ),
            cost_usd=0.0012,
        )
    if step == 1:
        return Completion(
            content="One of those is unknown. Converting the one that resolved.",
            tool_calls=(ToolCall(id="c", name="convert", arguments={"amount": "118.4", "rate": 0.091}),),
            cost_usd=0.0009,
        )
    return Completion(content="SE0000108656 is 118.40 SEK, about 10.77 USD.", cost_usd=0.0006)


if __name__ == "__main__":
    result = run(
        scripted_model,
        "What is SE0000108656 worth in dollars, and SE0000000000?",
        tools=[price, convert],
        budget=Budget(max_steps=6, max_cost_usd=0.05),
    )
    for step in result.steps:
        print(f"step {step.index}: {step.completion.content}")
        for call in step.tool_runs:
            mark = "x" if call.failed else "-"
            print(f"  {mark} {call.call.name}({json.dumps(call.call.arguments)}) -> {call.result}")
    print()
    print(f"stop={result.stop} tools={result.tool_calls} failed={result.failed_tool_calls} "
          f"cost=${result.cost_usd:.4f}")
