"""The parameterised strategy object.

A ``StrategySpec`` is the contract between every stage of the pipeline: Claude
produces one from a plain-English pitch, the signal engine evaluates it, the
simulator trades it, the sweeper perturbs it, and the IBKR layer deploys it.
It is pure data (JSON-serialisable) so a strategy can be re-tuned at any stage
without being rebuilt.

Every tunable number lives in ``params`` under an explicit name. Expressions
(``signal``, the entry/exit rules and the construction knobs) refer to those
names, so a sweep only ever has to override ``params``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, model_validator

Rebalance = Literal["daily", "weekly", "monthly"]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class Param(BaseModel):
    """One named, tunable knob."""

    name: str
    value: float
    low: float
    high: float
    kind: Literal["int", "float"] = "float"
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Param":
        if not _NAME_RE.match(self.name):
            raise ValueError(f"param name {self.name!r} must be snake_case")
        if self.low > self.high:
            raise ValueError(f"param {self.name}: low > high")
        return self

    def coerce(self, value: float) -> float:
        return float(int(round(value))) if self.kind == "int" else float(value)

    def grid(self, n: int = 3) -> list[float]:
        """``n`` evenly spaced values across [low, high], always including ``value``."""
        if n <= 1 or self.low == self.high:
            return [self.coerce(self.value)]
        pts = [self.low + i * (self.high - self.low) / (n - 1) for i in range(n)]
        vals = {self.coerce(p) for p in pts} | {self.coerce(self.value)}
        return sorted(vals)


class Universe(BaseModel):
    """What the strategy may trade.

    ``name`` is a point-in-time membership set in the warehouse (``"all"`` means
    every ticker with a price on the day). ``tickers`` optionally narrows it.
    """

    name: str = "all"
    tickers: list[str] = []
    min_adv: Optional[float] = None  # drop names whose 20-day average dollar volume is below this
    description: str = ""


class Rules(BaseModel):
    """Boolean entry/exit expressions for ``construction.mode == "rules"``.

    A position opens when its entry expression is true and stays open until its
    exit expression is true. Leave a side as ``None`` to never trade it.
    """

    entry_long: Optional[str] = None
    exit_long: Optional[str] = None
    entry_short: Optional[str] = None
    exit_short: Optional[str] = None


class Construction(BaseModel):
    """How scores/rules become portfolio weights.

    The numeric fields are scalar expressions over ``params`` (a literal such as
    ``"0.2"`` or a param name such as ``"top_quantile"``), so they are tunable
    through the same mechanism as the signal.
    """

    mode: Literal["quantile", "rules"] = "quantile"
    direction: Literal["long_short", "long_only", "short_only"] = "long_short"
    weighting: Literal["equal", "signal", "inverse_vol"] = "equal"
    long_quantile: str = "0.2"
    short_quantile: str = "0.2"
    max_weight: str = "0.1"
    gross_exposure: str = "1.0"
    vol_lookback: str = "60"


class StrategySpec(BaseModel):
    schema_version: int = 1
    name: str
    idea: str = ""
    description: str = ""
    hypothesis: str = ""
    universe: Universe = Universe()
    params: list[Param] = []
    signal: str
    rules: Optional[Rules] = None
    construction: Construction = Construction()
    rebalance: Rebalance = "weekly"
    assumptions: list[str] = []

    @model_validator(mode="after")
    def _check(self) -> "StrategySpec":
        names = [p.name for p in self.params]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate param names: {sorted(dupes)}")
        if self.construction.mode == "rules":
            r = self.rules
            if r is None or not (r.entry_long or r.entry_short):
                raise ValueError("construction.mode='rules' needs rules.entry_long or rules.entry_short")
        return self

    # -- knobs -----------------------------------------------------------
    def param_values(self) -> dict[str, float]:
        return {p.name: p.coerce(p.value) for p in self.params}

    def with_params(self, **overrides: Any) -> "StrategySpec":
        """Copy with some knobs changed. ``rebalance`` is accepted as a knob too."""
        spec = self.model_copy(deep=True)
        if "rebalance" in overrides:
            spec.rebalance = overrides.pop("rebalance")
        by_name = {p.name: p for p in spec.params}
        unknown = set(overrides) - set(by_name)
        if unknown:
            raise KeyError(f"unknown params {sorted(unknown)}; known: {sorted(by_name)}")
        for k, v in overrides.items():
            by_name[k].value = by_name[k].coerce(v)
        return StrategySpec.model_validate(spec.model_dump())

    def default_grid(self, n: int = 3) -> dict[str, list[float]]:
        return {p.name: p.grid(n) for p in self.params if p.low < p.high}

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StrategySpec":
        return cls.model_validate_json(Path(path).read_text())
