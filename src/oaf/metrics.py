"""Performance metrics.

The minimum reporting set is Sharpe, max drawdown, cumulative and annualised
return, Information Coefficient and Information Ratio. When a configuration has
been *selected* from many, the headline Sharpe is biased upwards, so we also
report the Probabilistic and Deflated Sharpe Ratios (Bailey & Lopez de Prado,
2012/2014).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

_N = NormalDist()
_EULER_GAMMA = 0.5772156649015329


# -- return-series metrics ---------------------------------------------------
def cumulative_return(r: pd.Series) -> float:
    return float((1 + r).prod() - 1)


def annualised_return(r: pd.Series, ppy: int = 252) -> float:
    if len(r) == 0:
        return float("nan")
    growth = float((1 + r).prod())
    return growth ** (ppy / len(r)) - 1 if growth > 0 else -1.0


def annualised_vol(r: pd.Series, ppy: int = 252) -> float:
    return float(r.std(ddof=1) * math.sqrt(ppy)) if len(r) > 1 else float("nan")


def sharpe_ratio(r: pd.Series, ppy: int = 252, rf: float = 0.0) -> float:
    """Annualised Sharpe of per-period returns; ``rf`` is an annual rate."""
    ex = r - rf / ppy
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(ppy)) if len(r) > 1 and sd > 0 else float("nan")


def sortino_ratio(r: pd.Series, ppy: int = 252) -> float:
    downside = math.sqrt(float((r.clip(upper=0) ** 2).mean()))
    return float(r.mean() / downside * math.sqrt(ppy)) if downside > 0 else float("nan")


def drawdown_series(r: pd.Series) -> pd.Series:
    equity = (1 + r).cumprod()
    return equity / equity.cummax().clip(lower=1.0) - 1


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough loss, as a negative number."""
    return float(drawdown_series(r).min()) if len(r) else float("nan")


def information_ratio(r: pd.Series, benchmark: pd.Series, ppy: int = 252) -> float:
    """Annualised mean active return over tracking error."""
    active = (r - benchmark.reindex(r.index).fillna(0.0)).dropna()
    te = active.std(ddof=1)
    return float(active.mean() / te * math.sqrt(ppy)) if len(active) > 1 and te > 0 else float("nan")


# -- signal metrics ------------------------------------------------------------
def forward_returns(asset_returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return earned over the ``horizon`` sessions *after* each date (starts at t+1)."""
    logr = np.log1p(asset_returns.clip(lower=-0.999))
    fwd = logr.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return np.expm1(fwd)


def information_coefficient(scores: pd.DataFrame, asset_returns: pd.DataFrame, horizon: int = 5) -> dict[str, float]:
    """Spearman rank IC between the signal and subsequent returns.

    ``ic_tstat`` uses non-overlapping windows (every ``horizon``-th date) so that
    overlapping forward returns do not overstate significance.
    """
    fwd = forward_returns(asset_returns, horizon)
    both = scores.notna() & fwd.notna()
    enough = both.sum(axis=1) >= 5
    a = scores.where(both).rank(axis=1)
    b = fwd.where(both).rank(axis=1)
    ic = a.corrwith(b, axis=1)[enough].dropna()
    if len(ic) < 2:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_ir": float("nan"), "ic_tstat": float("nan"), "ic_hit_rate": float("nan")}
    indep = ic.iloc[::horizon]
    sd = float(ic.std(ddof=1))
    t = float(indep.mean() / indep.std(ddof=1) * math.sqrt(len(indep))) if len(indep) > 1 and indep.std(ddof=1) > 0 else float("nan")
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": sd,
        "ic_ir": float(ic.mean() / sd) if sd > 0 else float("nan"),
        "ic_tstat": t,
        "ic_hit_rate": float((ic > 0).mean()),
    }


# -- selection-bias corrections --------------------------------------------------
def probabilistic_sharpe(r: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > ``benchmark_sr``), both per-period (not annualised).

    Accounts for track-record length and for the skew / fat tails of the returns.
    """
    r = r.dropna()
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or not sd > 0:
        return float("nan")
    sr = float(r.mean() / sd)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return float("nan")
    return _N.cdf((sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, trial_sharpe_var: float) -> float:
    """Expected best per-period Sharpe among ``n_trials`` strategies with zero true skill."""
    if n_trials < 2 or trial_sharpe_var <= 0:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(trial_sharpe_var) * ((1 - _EULER_GAMMA) * a + _EULER_GAMMA * b)


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sharpe_var: float) -> float:
    """PSR measured against the Sharpe you would expect from luck alone after ``n_trials`` tries.

    ``trial_sharpe_var`` is the variance of the per-period Sharpe ratios across the
    trials. Above ~0.95 the selected strategy is unlikely to be a fluke of the search.
    Trials that are highly correlated count for less than one each, so passing the raw
    number of configurations is conservative.
    """
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, trial_sharpe_var))


# -- one-stop summary ---------------------------------------------------------------
def summarize(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    scores: Optional[pd.DataFrame] = None,
    asset_returns: Optional[pd.DataFrame] = None,
    ic_horizon: int = 5,
    periods_per_year: int = 252,
    n_trials: int = 1,
    trial_sharpe_var: float = 0.0,
) -> dict[str, float]:
    ppy = periods_per_year
    mdd = max_drawdown(returns)
    ann = annualised_return(returns, ppy)
    out: dict[str, float] = {
        "sharpe": sharpe_ratio(returns, ppy),
        "max_drawdown": mdd,
        "cumulative_return": cumulative_return(returns),
        "annualised_return": ann,
        "annualised_vol": annualised_vol(returns, ppy),
        "sortino": sortino_ratio(returns, ppy),
        "calmar": ann / abs(mdd) if mdd < 0 else float("nan"),
        "hit_rate": float((returns > 0).mean()) if len(returns) else float("nan"),
    }
    if benchmark is not None:
        out["information_ratio"] = information_ratio(returns, benchmark, ppy)
        out["benchmark_annualised_return"] = annualised_return(benchmark.reindex(returns.index).fillna(0.0), ppy)
    if scores is not None and asset_returns is not None:
        out.update(information_coefficient(scores, asset_returns, ic_horizon))
    if turnover is not None and len(turnover):
        out["annual_turnover"] = float(turnover.mean() * ppy)
    out["psr"] = probabilistic_sharpe(returns)
    out["n_trials"] = n_trials
    out["deflated_sharpe"] = deflated_sharpe(returns, n_trials, trial_sharpe_var)
    return out
