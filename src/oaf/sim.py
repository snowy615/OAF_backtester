"""The internal simulated portfolio.

Two passes:

1. **Base book** - trades the strategy's target weights with realistic timing
   (decide at the close of day *d*, fill at the close of *d + execution_lag*,
   earn from the next day), lets weights drift between rebalances, charges
   commission + slippage on turnover and borrow on shorts, books delisting
   returns and moves the proceeds to cash.
2. **Risk overlay** - portfolio-level controls that sit above any one strategy:
   volatility targeting and a drawdown circuit breaker. Both scale the whole
   book using only returns already realised, and pay costs for the resizing.

Keeping risk controls here rather than in the signal means a strategy pitched
without them is still run under the fund's risk rules, and the same rules can
be swept like any other knob (``sim.vol_target``, ``sim.dd_limit``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from . import metrics as M
from .panel import Panel
from .signal import apply_universe, compute_scores, target_weights
from .spec import StrategySpec


class SimConfig(BaseModel):
    initial_capital: float = 1_000_000.0
    commission_bps: float = 1.0  # per unit of turnover, one way
    slippage_bps: float = 5.0  # per unit of turnover, one way
    borrow_bps_annual: float = 50.0  # charged on short notional
    execution_lag: int = 1  # sessions between the decision close and the fill close
    # portfolio-level risk overlay
    vol_target: Optional[float] = None  # annualised, e.g. 0.10
    vol_lookback: int = 60
    max_leverage: float = 2.0
    dd_limit: Optional[float] = None  # e.g. 0.15 -> go flat at -15% from peak
    dd_cooldown_days: int = 21
    ic_horizon: int = 5  # forward-return horizon for the information coefficient
    periods_per_year: int = 252

    @property
    def cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    spec: StrategySpec
    config: SimConfig
    returns: pd.Series  # net, after costs and risk overlay
    base_returns: pd.Series  # net of costs, before the risk overlay
    benchmark: pd.Series
    weights: pd.DataFrame  # held at each close, before overlay scaling
    turnover: pd.Series
    costs: pd.Series
    exposure_scale: pd.Series  # overlay multiplier applied to each day's return
    breaker_active: pd.Series
    metrics: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return self.config.initial_capital * (1 + self.returns).cumprod()


def rebalance_flags(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True on decision days: every day, or the last session of each week / month."""
    if freq == "daily":
        return np.ones(len(dates), dtype=bool)
    code = "W-FRI" if freq == "weekly" else "M"
    period = dates.to_period(code)
    # the final session is a period end only if the next business day starts a new period
    last = (dates[-1] + pd.offsets.BDay(1)).to_period(code) != period[-1]
    return np.asarray(np.r_[period[1:] != period[:-1], last], dtype=bool)


def equal_weight_benchmark(panel: Panel) -> pd.Series:
    """Equal-weight return of yesterday's universe members - the fair 'do nothing clever' book."""
    members = panel.mask.shift(1, fill_value=False)
    return panel.returns.where(members).mean(axis=1).fillna(0.0)


def simulate(
    targets: pd.DataFrame,
    panel: Panel,
    config: SimConfig = SimConfig(),
    rebalance: str = "weekly",
) -> dict[str, pd.Series | pd.DataFrame]:
    """Run the base book + risk overlay for a frame of target weights."""
    dates = panel.dates
    R = panel.returns.fillna(0.0).values
    W = targets.reindex(index=dates, columns=panel.tickers).values
    decided = ~np.isnan(W).all(axis=1)
    W = np.nan_to_num(W)
    tradable = panel.fields["close"].notna().values
    flags = rebalance_flags(dates, rebalance)

    T, N = R.shape
    # index of each ticker's final print; after its delisting return is booked the cash is freed
    has_px = tradable.any(axis=0)
    last_px = np.where(has_px, T - 1 - np.argmax(tradable[::-1], axis=0), -1)

    w = np.zeros(N)
    base = np.zeros(T)
    turnover = np.zeros(T)
    costs = np.zeros(T)
    held = np.zeros((T, N))
    borrow = config.borrow_bps_annual / 1e4 / config.periods_per_year
    lag = config.execution_lag

    for t in range(T):
        pnl = float(w @ R[t])
        carry = borrow * float(-w[w < 0].sum())
        if pnl > -1:
            w = w * (1 + R[t]) / (1 + pnl)
        w[t > last_px] = 0.0

        cost = 0.0
        src = t - lag
        if src >= 0 and flags[src] and decided[src]:
            tgt = np.where(tradable[t], W[src], w)  # cannot trade what has no price today
            turnover[t] = np.abs(tgt - w).sum()
            cost = turnover[t] * config.cost_rate
            w = tgt
        costs[t] = cost + carry
        base[t] = pnl - cost - carry
        held[t] = w

    base_s = pd.Series(base, index=dates)
    gross = np.abs(held).sum(axis=1)
    net, scale, active, overlay_cost = _risk_overlay(base, gross, config)
    return {
        "returns": pd.Series(net, index=dates),
        "base_returns": base_s,
        "weights": pd.DataFrame(held, index=dates, columns=panel.tickers),
        "turnover": pd.Series(turnover, index=dates),
        "costs": pd.Series(costs + overlay_cost, index=dates),
        "exposure_scale": pd.Series(scale, index=dates),
        "breaker_active": pd.Series(active, index=dates),
    }


def _risk_overlay(base: np.ndarray, gross: np.ndarray, cfg: SimConfig):
    """Vol targeting + drawdown circuit breaker on the whole book.

    The multiplier applied to day *t* is fixed at the close of *t - 1* from returns
    realised up to then. When the breaker trips the book is flat for
    ``dd_cooldown_days``; on re-entry the high-water mark resets to current equity so
    the book is not immediately stopped out again.
    """
    T = len(base)
    applied = np.ones(T)
    active = np.zeros(T, dtype=bool)
    cost = np.zeros(T)
    if cfg.vol_target is None and cfg.dd_limit is None:
        return base.copy(), applied, active, cost

    net = np.zeros(T)
    ann = np.sqrt(cfg.periods_per_year)
    equity = peak = 1.0
    cooldown = 0
    next_scale = prev_scale = 1.0
    for t in range(T):
        s = next_scale
        applied[t] = s
        cost[t] = abs(s - prev_scale) * (gross[t - 1] if t else 0.0) * cfg.cost_rate
        net[t] = s * base[t] - cost[t]
        equity *= 1 + net[t]
        peak = max(peak, equity)
        prev_scale = s

        if cooldown > 0:
            cooldown -= 1
            if cooldown == 0:
                peak = equity
        elif cfg.dd_limit is not None and equity / peak - 1 <= -cfg.dd_limit:
            cooldown = cfg.dd_cooldown_days
        active[t] = cooldown > 0

        target = 1.0
        if cfg.vol_target is not None and t + 1 >= cfg.vol_lookback:
            window = base[t + 1 - cfg.vol_lookback : t + 1]
            window = window[window != 0]  # ignore sessions before the book was invested
            if len(window) >= max(10, cfg.vol_lookback // 2) and window.std(ddof=1) > 0:
                target = min(cfg.max_leverage, cfg.vol_target / (window.std(ddof=1) * ann))
        next_scale = 0.0 if cooldown > 0 else target
    return net, applied, active, cost


def run_backtest(
    spec: StrategySpec,
    panel: Panel,
    config: SimConfig = SimConfig(),
    n_trials: int = 1,
    trial_sharpe_var: float = 0.0,
) -> BacktestResult:
    """Spec -> scores -> weights -> simulated book -> metrics.

    ``n_trials`` / ``trial_sharpe_var`` describe how many configurations this one was
    picked from, so the deflated Sharpe can correct for the selection.
    """
    panel = apply_universe(spec, panel)
    scores = compute_scores(spec, panel)
    targets = target_weights(spec, panel, scores)
    sim = simulate(targets, panel, config, spec.rebalance)
    benchmark = equal_weight_benchmark(panel)
    result = BacktestResult(spec=spec, config=config, benchmark=benchmark, **sim)

    live = sim["weights"].abs().sum(axis=1) > 0
    start = live.idxmax() if live.any() else panel.dates[0]
    result.metrics = M.summarize(
        returns=result.returns.loc[start:],
        benchmark=benchmark.loc[start:],
        turnover=result.turnover.loc[start:],
        scores=scores.loc[start:],
        asset_returns=panel.returns.loc[start:],
        ic_horizon=config.ic_horizon,
        periods_per_year=config.periods_per_year,
        n_trials=n_trials,
        trial_sharpe_var=trial_sharpe_var,
    )
    result.metrics["start"] = str(pd.Timestamp(start).date())
    result.metrics["end"] = str(panel.dates[-1].date())
    return result
