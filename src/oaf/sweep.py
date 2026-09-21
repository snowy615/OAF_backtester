"""Systematic parameter tuning: grid sweeps and walk-forward validation.

A knob is addressed by name:

- a strategy param            ``lookback=20,60,120``
- the rebalance frequency     ``rebalance=weekly,monthly``
- a portfolio-sim setting     ``sim.vol_target=0.08,0.12``

A sweep picks a best configuration *in sample*, which flatters it; the result
therefore carries a deflated Sharpe that accounts for the number of trials.
Prefer :func:`walk_forward`, which re-picks the configuration on each training
window and only ever scores it on the following, unseen window.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from . import metrics as M
from .panel import Panel
from .sim import BacktestResult, SimConfig, equal_weight_benchmark, run_backtest
from .spec import StrategySpec

Overrides = dict[str, Any]

OBJECTIVES: dict[str, Callable[[pd.Series, int], float]] = {
    "sharpe": M.sharpe_ratio,
    "sortino": M.sortino_ratio,
    "annualised_return": M.annualised_return,
    "calmar": lambda r, ppy: M.annualised_return(r, ppy) / abs(M.max_drawdown(r)) if M.max_drawdown(r) < 0 else np.nan,
}


def apply_overrides(spec: StrategySpec, config: SimConfig, overrides: Overrides) -> tuple[StrategySpec, SimConfig]:
    sim_kw = {k[4:]: v for k, v in overrides.items() if k.startswith("sim.")}
    spec_kw = {k: v for k, v in overrides.items() if not k.startswith("sim.")}
    unknown = set(sim_kw) - set(SimConfig.model_fields)
    if unknown:
        raise KeyError(f"unknown sim settings {sorted(unknown)}")
    new_config = SimConfig.model_validate({**config.model_dump(), **sim_kw}) if sim_kw else config
    return (spec.with_params(**spec_kw) if spec_kw else spec), new_config


def expand_grid(grid: dict[str, list], max_trials: Optional[int] = None, seed: int = 0) -> list[Overrides]:
    """Cartesian product of the grid; a seeded random subset of it if larger than ``max_trials``."""
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    if max_trials is not None and len(combos) > max_trials:
        combos = random.Random(seed).sample(combos, max_trials)
    return combos


def parse_grid(items: list[str]) -> dict[str, list]:
    """``["lookback=20,60", "rebalance=weekly,monthly"]`` -> grid dict."""
    grid: dict[str, list] = {}
    for item in items:
        key, _, raw = item.partition("=")
        if not raw:
            raise ValueError(f"bad grid item {item!r}; expected name=v1,v2,...")
        grid[key.strip()] = [_parse_value(v) for v in raw.split(",")]
    return grid


def _parse_value(v: str) -> Any:
    v = v.strip()
    if v.lower() in ("none", "null"):
        return None
    try:
        return float(v)
    except ValueError:
        return v


@dataclass
class SweepResult:
    table: pd.DataFrame  # one row per trial: knobs + metrics, best first
    returns: pd.DataFrame  # dates x trial id, net returns
    trials: list[Overrides]
    objective: str
    best: BacktestResult  # re-scored with the deflated Sharpe for len(trials) tries

    @property
    def best_overrides(self) -> Overrides:
        return self.trials[int(self.table.index[0])]


def _run_trials(spec, panel, trials, config, progress) -> tuple[list[BacktestResult], pd.DataFrame]:
    results = []
    for i, ov in enumerate(trials):
        s, c = apply_overrides(spec, config, ov)
        results.append(run_backtest(s, panel, c))
        if progress:
            progress(i + 1, len(trials), ov, results[-1])
    returns = pd.DataFrame({i: r.returns for i, r in enumerate(results)})
    return results, returns


def run_sweep(
    spec: StrategySpec,
    panel: Panel,
    grid: Optional[dict[str, list]] = None,
    config: SimConfig = SimConfig(),
    objective: str = "sharpe",
    max_trials: Optional[int] = None,
    seed: int = 0,
    progress: Optional[Callable] = None,
) -> SweepResult:
    grid = grid or spec.default_grid()
    trials = expand_grid(grid, max_trials, seed)
    if not trials:
        raise ValueError("empty grid")
    results, returns = _run_trials(spec, panel, trials, config, progress)

    rows = [{**ov, **r.metrics} for ov, r in zip(trials, results)]
    score_key = objective if objective in rows[0] else "sharpe"
    table = pd.DataFrame(rows).sort_values(score_key, ascending=False, na_position="last")

    ppy = config.periods_per_year
    sharpe_var = float(np.nanvar(table["sharpe"].values / np.sqrt(ppy), ddof=1)) if len(trials) > 1 else 0.0
    best_spec, best_cfg = apply_overrides(spec, config, trials[int(table.index[0])])
    best = run_backtest(best_spec, panel, best_cfg, n_trials=len(trials), trial_sharpe_var=sharpe_var)
    return SweepResult(table=table, returns=returns, trials=trials, objective=objective, best=best)


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series  # stitched out-of-sample returns
    folds: pd.DataFrame  # per fold: windows, chosen knobs, in-sample vs out-of-sample score
    metrics: dict[str, float]
    trials: list[Overrides]
    objective: str

    @property
    def degradation(self) -> float:
        """Mean out-of-sample minus in-sample objective. Strongly negative = overfit."""
        return float((self.folds["oos_score"] - self.folds["is_score"]).mean())


def walk_forward(
    spec: StrategySpec,
    panel: Panel,
    grid: Optional[dict[str, list]] = None,
    config: SimConfig = SimConfig(),
    train_days: int = 756,
    test_days: int = 252,
    expanding: bool = False,
    objective: str = "sharpe",
    max_trials: Optional[int] = None,
    seed: int = 0,
    progress: Optional[Callable] = None,
) -> WalkForwardResult:
    """Pick the best configuration on each training window, score it on the next window only.

    Signals and the simulator are strictly backward-looking, so each trial is simulated
    once over the full history and the windows are read off its return series. (The risk
    overlay's state carries across window boundaries, exactly as it would live.)
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
    score = OBJECTIVES[objective]
    ppy = config.periods_per_year
    grid = grid or spec.default_grid()
    trials = expand_grid(grid, max_trials, seed)
    results, returns = _run_trials(spec, panel, trials, config, progress)

    # start once every trial is actually trading, so slow lookbacks are not penalised
    live = [r.weights.abs().sum(axis=1).gt(0).values.argmax() for r in results]
    first = int(max(live))
    n = len(returns)
    if first + train_days + test_days > n:
        raise ValueError(
            f"not enough history: need {train_days}+{test_days} sessions after warm-up, have {n - first}"
        )

    fold_rows, pieces = [], []
    test_start = first + train_days
    while test_start < n:
        train_start = first if expanding else test_start - train_days
        test_end = min(test_start + test_days, n)
        train = returns.iloc[train_start:test_start]
        test = returns.iloc[test_start:test_end]
        is_scores = train.apply(lambda col: score(col, ppy))
        if is_scores.notna().any():
            pick = int(is_scores.idxmax())
            pieces.append(test[pick])
            fold_rows.append(
                {
                    "train_start": train.index[0].date(),
                    "test_start": test.index[0].date(),
                    "test_end": test.index[-1].date(),
                    "trial": pick,
                    **trials[pick],
                    "is_score": float(is_scores[pick]),
                    "oos_score": score(test[pick], ppy),
                }
            )
        test_start = test_end

    oos = pd.concat(pieces)
    benchmark = equal_weight_benchmark(panel)
    summary = M.summarize(oos, benchmark=benchmark.reindex(oos.index), periods_per_year=ppy)
    # out-of-sample returns were never selected on, so no deflation is needed
    summary.pop("deflated_sharpe", None)
    summary["n_trials"] = len(trials)
    summary["n_folds"] = len(fold_rows)
    return WalkForwardResult(oos_returns=oos, folds=pd.DataFrame(fold_rows), metrics=summary, trials=trials, objective=objective)
