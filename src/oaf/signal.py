"""StrategySpec + Panel -> scores -> target weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import dsl
from .panel import Panel
from .spec import StrategySpec


def validate_spec(spec: StrategySpec, fields: list[str], groups: list[str] | None = None) -> list[str]:
    """Static checks that need no data. Returns a list of problems (empty = valid)."""
    problems: list[str] = []
    params = spec.param_values()
    known = set(fields) | set(params)

    exprs = {"signal": spec.signal}
    if spec.rules:
        exprs.update({f"rules.{k}": v for k, v in spec.rules.model_dump().items() if v})
    used: set[str] = set()
    for where, expr in exprs.items():
        try:
            names = dsl.names_in(expr)
        except dsl.DSLError as e:
            problems.append(f"{where}: {e}")
            continue
        used |= names
        for n in sorted(names - known):
            problems.append(f"{where}: unknown name {n!r} (not a param or an available field)")

    c = spec.construction
    for knob in ("long_quantile", "short_quantile", "max_weight", "gross_exposure", "vol_lookback"):
        expr = getattr(c, knob)
        try:
            used |= dsl.names_in(expr)
            dsl.evaluate_scalar(expr, params)
        except dsl.DSLError as e:
            problems.append(f"construction.{knob}: {e}")

    for p in spec.params:
        if not p.low <= p.value <= p.high:
            problems.append(f"param {p.name}: value {p.value} outside [{p.low}, {p.high}]")
        if p.name in fields:
            problems.append(f"param {p.name}: name collides with a data field")
        if p.name not in used:
            problems.append(f"param {p.name}: declared but never used")
    return problems


def compute_scores(spec: StrategySpec, panel: Panel) -> pd.DataFrame:
    """The signal: higher score = more attractive to hold long. NaN outside the universe."""
    return dsl.evaluate(spec.signal, panel, spec.param_values())


def target_weights(spec: StrategySpec, panel: Panel, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Desired weights for every date, using only information up to that date.

    Rows where nothing is tradable yet (indicator warm-up) are all-NaN, which the
    simulator reads as "no decision - keep what you hold".
    """
    params = spec.param_values()
    c = spec.construction
    scalar = lambda name: dsl.evaluate_scalar(getattr(c, name), params)  # noqa: E731
    gross, cap = scalar("gross_exposure"), scalar("max_weight")
    if scores is None:
        scores = compute_scores(spec, panel)

    if c.mode == "rules":
        side = _rule_positions(spec, panel)
        long_mask, short_mask = side > 0, side < 0
    else:
        q_long, q_short = scalar("long_quantile"), scalar("short_quantile")
        if not (0 < q_long <= 1 and 0 < q_short <= 1):
            raise ValueError("long_quantile / short_quantile must be in (0, 1]")
        pct = scores.rank(axis=1, pct=True)
        n = scores.notna().sum(axis=1)
        # at least one name per leg, and the legs never overlap
        long_mask = pct.gt(1 - np.maximum(q_long, 1 / n.clip(lower=1)), axis=0)
        short_mask = pct.le(np.maximum(q_short, 1 / n.clip(lower=1)), axis=0) & ~long_mask

    if c.direction == "long_only":
        short_mask = short_mask & False
    elif c.direction == "short_only":
        long_mask = long_mask & False
    both = c.direction == "long_short"
    leg_gross = gross / 2 if both else gross

    if c.weighting == "signal":
        mid = scores.median(axis=1)
        raw = scores.sub(mid, axis=0).abs() + 1e-12
    elif c.weighting == "inverse_vol":
        vol = panel.returns.rolling(int(round(scalar("vol_lookback")))).std()
        raw = 1.0 / vol.replace(0, np.nan)
    else:
        raw = pd.DataFrame(1.0, index=scores.index, columns=scores.columns)

    # In rules mode a lone position must not be levered up to fill the book.
    fill = c.mode == "quantile"
    longs = _leg(raw.where(long_mask), leg_gross, cap, fill).fillna(0.0)
    shorts = _leg(raw.where(short_mask), leg_gross, cap, fill).fillna(0.0)
    weights = longs - shorts
    # rules are "decided" once the signal has warmed up, even if the decision is to be flat
    decided = (long_mask | short_mask).any(axis=1) if c.mode == "quantile" else scores.notna().any(axis=1)
    weights[~decided] = np.nan
    return weights


def _leg(raw: pd.DataFrame, leg_gross: float, cap: float, fill: bool) -> pd.DataFrame:
    """Normalise one leg to ``leg_gross`` with a per-name cap (excess is redistributed)."""
    w = raw.div(raw.sum(axis=1), axis=0) * leg_gross
    if not fill:
        return w.clip(upper=cap)
    for _ in range(8):
        over = w > cap
        if not over.any().any():
            break
        capped_sum = over.sum(axis=1) * cap
        free = w.where(~over)
        room = (leg_gross - capped_sum).clip(lower=0)
        free = free.div(free.sum(axis=1), axis=0).mul(room, axis=0)
        w = free.where(~over, cap)
    return w.clip(upper=cap)


def _rule_positions(spec: StrategySpec, panel: Panel) -> pd.DataFrame:
    """Walk the entry/exit state machine forward one day at a time: +1 long, -1 short, 0 flat."""
    params, r = spec.param_values(), spec.rules
    shape = panel.mask.shape
    cond = lambda e: (  # noqa: E731
        dsl.evaluate_condition(e, panel, params).values if e else np.zeros(shape, dtype=bool)
    )
    enter_l, exit_l, enter_s, exit_s = cond(r.entry_long), cond(r.exit_long), cond(r.entry_short), cond(r.exit_short)
    member = panel.mask.values
    state = np.zeros(shape[1], dtype=int)
    out = np.zeros(shape, dtype=int)
    for t in range(shape[0]):
        state[(state == 1) & exit_l[t]] = 0
        state[(state == -1) & exit_s[t]] = 0
        state[~member[t]] = 0  # dropped from the universe -> position closed
        flat = state == 0
        state[flat & enter_l[t]] = 1
        state[flat & enter_s[t] & ~enter_l[t]] = -1
        out[t] = state
    return pd.DataFrame(out, index=panel.dates, columns=panel.tickers)

