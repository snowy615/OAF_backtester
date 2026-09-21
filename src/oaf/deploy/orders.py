"""Broker-independent deployment maths: strategy -> target shares -> orders.

Everything here is pure, so a deployment can be planned, reviewed and unit-tested
without a broker connection. ``oaf.deploy.ibkr`` only executes what this produces.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel

from ..panel import Panel
from ..signal import target_weights
from ..sim import SimConfig, rebalance_flags, run_backtest
from ..spec import StrategySpec


class Order(BaseModel):
    ticker: str
    side: Literal["BUY", "SELL"]
    quantity: int
    est_price: float
    est_notional: float


class DeploymentPlan(BaseModel):
    strategy: str
    mode: Literal["paper", "live"]
    capital: float
    as_of: str  # last data date the targets are based on
    decision_date: str  # rebalance date whose signal set the targets
    exposure_scale: float  # risk-overlay multiplier (0 = circuit breaker has the book flat)
    target_weights: dict[str, float]
    target_shares: dict[str, int]
    current_shares: dict[str, int] = {}
    orders: list[Order] = []
    notes: list[str] = []

    def summary(self) -> str:
        gross = sum(abs(o.est_notional) for o in self.orders)
        lines = [
            f"{self.strategy} [{self.mode.upper()}] capital {self.capital:,.0f}",
            f"data as of {self.as_of}, signal from {self.decision_date}, overlay scale {self.exposure_scale:.2f}",
            f"{len(self.target_shares)} target positions, {len(self.orders)} orders, {gross:,.0f} notional to trade",
        ]
        lines += [f"  {o.side:<4} {o.quantity:>7} {o.ticker:<8} ~{o.est_notional:>12,.0f}" for o in self.orders]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2) + "\n")
        return path


def target_shares(weights: pd.Series, prices: pd.Series, capital: float) -> pd.Series:
    """Whole shares per ticker, rounded towards zero so the book never exceeds its allocation."""
    notional = weights * capital
    shares = (notional / prices).replace([math.inf, -math.inf], float("nan")).dropna()
    return shares.apply(math.trunc).astype(int).loc[lambda s: s != 0]


def diff_orders(current: dict[str, int], target: dict[str, int], prices: pd.Series, min_notional: float = 0.0) -> list[Order]:
    """Orders that move ``current`` holdings to ``target``. Sells first, so buys are funded."""
    orders = []
    for ticker in sorted(set(current) | set(target)):
        delta = int(target.get(ticker, 0)) - int(current.get(ticker, 0))
        price = float(prices.get(ticker, float("nan")))
        if delta == 0 or math.isnan(price) or abs(delta) * price < min_notional:
            continue
        orders.append(
            Order(ticker=ticker, side="BUY" if delta > 0 else "SELL", quantity=abs(delta), est_price=price, est_notional=abs(delta) * price)
        )
    return sorted(orders, key=lambda o: (o.side == "BUY", o.ticker))


def build_plan(
    spec: StrategySpec,
    panel: Panel,
    capital: float,
    config: SimConfig = SimConfig(),
    mode: Literal["paper", "live"] = "paper",
    current: Optional[dict[str, int]] = None,
    min_notional: float = 200.0,
) -> DeploymentPlan:
    """Size the strategy's latest targets to ``capital`` and diff against current holdings.

    The targets are exactly what the simulator would hold next: weights from the most
    recent rebalance decision, scaled by the risk overlay's current multiplier.
    """
    if spec.universe.tickers:
        panel = panel.restrict(spec.universe.tickers)
    result = run_backtest(spec, panel, config)
    targets = target_weights(spec, panel)
    flags = pd.Series(rebalance_flags(targets.index, spec.rebalance), index=targets.index)
    decided = targets.notna().any(axis=1) & flags
    if not decided.any():
        raise ValueError("the strategy has not produced a rebalance decision yet (not enough history?)")
    decision_date = decided[decided].index[-1]
    weights = targets.loc[decision_date].fillna(0.0)
    weights = weights[weights != 0]

    scale = float(result.exposure_scale.iloc[-1])
    notes = []
    if result.breaker_active.iloc[-1]:
        scale = 0.0
        notes.append("drawdown circuit breaker is active: target book is flat")
    prices = panel.fields.get("raw_close", panel.fields["close"]).ffill().iloc[-1]
    unpriced = [t for t in weights.index if pd.isna(prices.get(t))]
    if unpriced:
        notes.append(f"no price for {unpriced}; skipped")

    shares = target_shares(weights * scale, prices, capital)
    current = {k: int(v) for k, v in (current or {}).items()}
    foreign = sorted(set(current) - set(panel.tickers))
    if foreign:
        notes.append(f"positions outside the strategy's data left untouched: {foreign}")
    managed = {k: v for k, v in current.items() if k in set(panel.tickers)}
    tgt = {k: int(v) for k, v in shares.items()}
    return DeploymentPlan(
        strategy=spec.name,
        mode=mode,
        capital=capital,
        as_of=str(panel.dates[-1].date()),
        decision_date=str(decision_date.date()),
        exposure_scale=scale,
        target_weights={k: float(v) for k, v in weights.items()},
        target_shares=tgt,
        current_shares=managed,
        orders=diff_orders(managed, tgt, prices, min_notional),
        notes=notes,
    )
