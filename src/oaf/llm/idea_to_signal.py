"""Claude interface: plain-English idea -> parameterised ``StrategySpec``.

Claude writes the spec; the platform checks it. Every draft is validated against
the real DSL and the fields actually in the warehouse, and any problems are sent
back for repair, so what comes out of this module always runs.
"""

from __future__ import annotations

import json
from typing import Optional

from .. import dsl
from ..signal import validate_spec
from ..spec import StrategySpec
from .client import LLMError, get_client, parse_structured

SYSTEM_TEMPLATE = """\
You are the strategy structurer for the Oxford Alpha Fund's backtesting platform. Members \
pitch trading ideas in plain English; you turn each pitch into a precise, rules-based, \
fully parameterised strategy specification that the platform can backtest, tune, and later \
trade through Interactive Brokers.

# What you produce
A StrategySpec. Its parts:

- `name`: short snake_case identifier. `description`: two or three sentences on what the \
strategy does. `hypothesis`: why it should make money - the economic or behavioural reason \
the effect exists and who is on the other side. `idea`: the member's pitch, verbatim.
- `signal`: one expression in the platform DSL (below) giving each stock a score on each \
day. Higher score = more attractive to hold long. It must be a per-stock quantity, not a constant.
- `params`: every tunable number in the strategy, as a named knob with a sensible default \
`value`, a plausible `low`-`high` range to sweep, `kind` ("int" for lookbacks and day \
counts, "float" otherwise) and a one-line `description`. The point of the platform is that \
members can tune and stress every assumption, so numbers such as lookbacks, thresholds and \
quantiles belong in `params` and are referenced by name in expressions, rather than being \
written as literals. Structural constants (the 1 in `x / ts_delay(x, n) - 1`) stay literal. \
Only declare params you actually use.
- `construction`: how scores become a portfolio.
  - `mode` "quantile": each rebalance, go long the top `long_quantile` fraction of the \
universe by score and short the bottom `short_quantile`. Right for cross-sectional ideas \
("buy the cheapest", "winners beat losers").
  - `mode` "rules": positions open when an entry condition is true and stay open until an \
exit condition is true, per stock. Right for event/threshold ideas ("buy when RSI < 30, \
sell when it recovers above 50"). Put the conditions in `rules` (`entry_long`, `exit_long`, \
`entry_short`, `exit_short`; leave a side null to never trade it). Conditions are boolean \
DSL expressions.
  - `direction`: "long_short", "long_only" or "short_only". `weighting`: "equal", "signal" \
(proportional to score distance from the median) or "inverse_vol".
  - `long_quantile`, `short_quantile`, `max_weight`, `gross_exposure`, `vol_lookback` are \
strings holding either a number or the name of a param, so they are tunable too.
- `rebalance`: "daily", "weekly" or "monthly". Match it to how fast the signal changes; \
slow signals rebalanced daily just pay costs.
- `universe`: `name` must be one of the universes listed below. Leave `tickers` empty \
unless the pitch names specific stocks. When the universe is "all" (the whole market), set \
`min_adv` to a dollar-volume floor (e.g. 5000000 for $5m/day) so illiquid names are excluded; \
mention the floor in `assumptions`.
- `assumptions`: every judgement call you made where the pitch was silent or ambiguous, \
one per line, so the member can see and challenge them.

# Platform DSL
Names in an expression are either a param you declared or one of the data fields listed \
below. Arithmetic (+ - * / **), comparisons, `and` / `or` / `not`, and `a if cond else b` \
work element-wise. `ts_` operators look back over each stock's own history; `rank`, \
`zscore`, `demean`, `scale`, `winsorize` and `group_neutralize` work across the universe \
on a single day. Windows are in trading days. Everything is backward-looking by \
construction - there is no way to reference the future, and no need to lag the signal \
yourself: the simulator already trades one session after the signal is observed.

Operators:
{operators}

# Risk controls are not your job
Volatility targeting, leverage limits and drawdown circuit breakers are applied by the \
portfolio simulator to every strategy. Leave them out of the signal, and mention in \
`assumptions` if the pitch asked for one so the member knows to set it on the simulator.

# Judgement
Stay faithful to the pitch: formalise the member's idea rather than substituting a better \
one. Where the pitch is vague, choose the conventional reading from the literature and \
record it in `assumptions`. If the pitch needs data that is not in the field list, build \
the closest honest proxy from what is available and say so in `assumptions`; if no honest \
proxy exists, say that plainly in `description` and `assumptions` and give the nearest \
testable version. Prefer simple, robust expressions - ranks over raw values when combining \
quantities on different scales, and a skip window for momentum-style signals that would \
otherwise pick up short-term reversal.
"""


def build_system_prompt() -> str:
    return SYSTEM_TEMPLATE.format(operators=dsl.operator_docs())


def _context(fields: list[str], universes: list[str], groups: list[str]) -> str:
    return (
        f"Available data fields: {', '.join(fields)}\n"
        f"Available universes: {', '.join(universes)}\n"
        f"Available groups for group_neutralize: {', '.join(groups) or 'none'}"
    )


def _converse(client, messages: list[dict], fields: list[str], groups: list[str], max_repairs: int) -> StrategySpec:
    system = build_system_prompt()
    for attempt in range(max_repairs + 1):
        spec, response = parse_structured(client, system, messages, StrategySpec)
        problems = validate_spec(spec, fields, groups)
        if not problems:
            return spec
        if attempt == max_repairs:
            raise LLMError("Claude's strategy still fails validation:\n- " + "\n- ".join(problems))
        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": "The platform's validator rejected that spec:\n- "
                + "\n- ".join(problems)
                + "\nPlease return a corrected StrategySpec.",
            }
        )
    raise AssertionError("unreachable")


def structure_idea(
    idea: str,
    fields: list[str],
    universes: Optional[list[str]] = None,
    groups: Optional[list[str]] = None,
    client=None,
    max_repairs: int = 2,
) -> StrategySpec:
    """Turn a plain-English pitch into a validated ``StrategySpec``."""
    client = client or get_client()
    universes, groups = universes or ["all"], groups or []
    messages = [{"role": "user", "content": f"{_context(fields, universes, groups)}\n\nPitch:\n{idea.strip()}"}]
    spec = _converse(client, messages, fields, groups, max_repairs)
    spec.idea = idea.strip()
    return spec


def refine_spec(
    spec: StrategySpec,
    feedback: str,
    fields: list[str],
    universes: Optional[list[str]] = None,
    groups: Optional[list[str]] = None,
    client=None,
    max_repairs: int = 2,
) -> StrategySpec:
    """Loop back: revise an existing spec from feedback (e.g. after seeing backtest results)."""
    client = client or get_client()
    universes, groups = universes or ["all"], groups or []
    content = (
        f"{_context(fields, universes, groups)}\n\n"
        f"Current StrategySpec:\n{json.dumps(spec.model_dump(), indent=2)}\n\n"
        f"Requested change:\n{feedback.strip()}\n\n"
        "Return the full revised StrategySpec. Keep everything the request does not touch, "
        "including param names, so earlier sweeps stay comparable."
    )
    new = _converse(client, [{"role": "user", "content": content}], fields, groups, max_repairs)
    new.idea = spec.idea
    return new
