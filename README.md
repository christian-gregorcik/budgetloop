# toolloop

The tool-calling loop, written out: schemas from type hints, parallel calls, budgets
and a trace. No dependencies.

Every agent framework hides the same forty lines. This is those lines, with the parts
that actually bite — a tool that raises, arguments that nearly fit, a model that never
stops asking — handled where you can see them.

## The shape of it

```python
from toolloop import Budget, tool, run


@tool
def price(isin: str) -> float:
    """Look up the last price for an instrument.

    Args:
        isin: the twelve-character identifier
    """
    return lookup(isin)


result = run(
    complete,                      # your model, see below
    "What is SE0000108656 worth?",
    tools=[price],
    budget=Budget(max_steps=6, max_cost_usd=0.05),
)

print(result.answer, result.stop, result.cost_usd)
```

## The model

One callable, provider-agnostic:

```python
def complete(messages: list[Message], tools: list[dict]) -> Completion:
    ...
```

It gets the conversation and the tool schemas, and returns either text or tool calls.
Adapting an SDK to it is a dozen lines, and in return the whole loop is testable with a
scripted model and no network. Every test in this repository runs offline.

## Run the example

```
python examples/offline.py
```

```
step 0: Looking both of them up.
  - price({"isin": "SE0000108656"}) -> 118.4
  x price({"isin": "SE0000000000"}) -> error: LookupError: SE0000000000 is not in the price list
step 1: One of those is unknown. Converting the one that resolved.
  - convert({"amount": "118.4", "rate": 0.091}) -> 10.77
step 2: SE0000108656 is 118.40 SEK, about 10.77 USD.

stop=answered tools=3 failed=1 cost=$0.0027
```

Three things are visible in that output, and each one is a decision.

**A tool that raises becomes a result, not a crash.** The second lookup failed, the
model read the failure and worked around it. A loop that propagates the exception
throws away a model that was about to recover on its own.

**Arguments are checked, and corrected once where the intent is not in doubt.** The
model sent `"118.4"` as a string. A number written as a string is the mistake models
make constantly, so it is coerced; `"five"` is not, and comes back as
`error: convert.amount: expected number, got str`. Unknown argument, missing argument,
value outside an enum: same treatment, reported to the model in words it can act on.

**Both calls in step 0 ran at the same time.** When a model asks for several tools in
one turn it means they are independent, so they are run on a pool and the results go
back in the order they were asked for.

## Schemas come from the signature

`@tool` builds the JSON Schema from the type hints and the docstring. Writing the
schema by hand next to the function is how the two drift apart, and a drifted schema
shows up as a model calling the tool wrongly for reasons nobody can see.

Supported: `str`, `int`, `float`, `bool`, `list[T]`, `dict[str, T]`, `Literal[...]`,
unions with `None`, `Any`, and defaults. A parameter without a type hint is an error at
import time, not a surprise at runtime. `Args:` lines in the docstring become parameter
descriptions, which is most of what makes a model call a tool correctly.

## Budgets

```python
Budget(max_steps=10, max_seconds=30, max_cost_usd=0.50, max_tool_calls=25)
```

Only `max_steps` has a default, because a loop without a ceiling is a loop that bills
you for a model's bad afternoon. A run that hits a limit ends with `result.stop` set to
which one, and `result.answered` is `False`. Nothing raises.

## The trace

`result.to_dict()` is plain data: every step, what the model said, what each tool was
called with, what came back, how long it took and what it cost. Store it next to the
request and a bad answer stops being a mystery.

## Tests

```
pytest -q
ruff check .
```

65 tests, no network, no API key. MIT licensed.
