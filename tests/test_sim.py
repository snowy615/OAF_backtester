import numpy as np
import pandas as pd
import pytest

from oaf.sim import SimConfig, _risk_overlay, rebalance_flags, run_backtest, simulate
from tests.conftest import make_panel

FREE = dict(commission_bps=0, slippage_bps=0, borrow_bps_annual=0)


def _one_asset(prices):
    dates = pd.bdate_range("2024-01-01", periods=len(prices))
    return make_panel(pd.DataFrame({"A": prices}, index=dates, dtype=float))


def test_execution_timing_decide_fill_earn():
    panel = _one_asset([100, 110, 121, 133.1, 146.41])  # +10% every day
    targets = pd.DataFrame(1.0, index=panel.dates, columns=["A"])
    out = simulate(targets, panel, SimConfig(execution_lag=1, **FREE), "daily")
    # decided at close 0, filled at close 1, so the first return earned is day 2's
    assert out["returns"].tolist() == pytest.approx([0, 0, 0.1, 0.1, 0.1])
    same_close = simulate(targets, panel, SimConfig(execution_lag=0, **FREE), "daily")
    assert same_close["returns"].tolist() == pytest.approx([0, 0.1, 0.1, 0.1, 0.1])


def test_costs_are_charged_on_turnover():
    panel = _one_asset([100, 100, 100, 100])
    targets = pd.DataFrame(1.0, index=panel.dates, columns=["A"])
    out = simulate(targets, panel, SimConfig(commission_bps=1, slippage_bps=5, borrow_bps_annual=0), "daily")
    assert out["turnover"].tolist() == pytest.approx([0, 1, 0, 0])
    assert out["returns"].iloc[1] == pytest.approx(-0.0006)
    assert out["returns"].iloc[2:].abs().sum() == 0


def test_short_borrow_cost():
    panel = _one_asset([100] * 5)
    targets = pd.DataFrame(-1.0, index=panel.dates, columns=["A"])
    out = simulate(targets, panel, SimConfig(commission_bps=0, slippage_bps=0, borrow_bps_annual=252), "daily")
    assert out["returns"].iloc[-1] == pytest.approx(-0.0001)


def test_weights_drift_between_rebalances():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [100, 100, 200, 200], "B": [100, 100, 100, 100]}, index=dates, dtype=float)
    panel = make_panel(close)
    targets = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    targets.iloc[0] = [0.5, 0.5]  # a single decision, then hold
    out = simulate(targets, panel, SimConfig(**FREE), "daily")
    assert out["weights"].iloc[2].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert out["returns"].iloc[2] == pytest.approx(0.5)


def test_delisting_return_is_booked_and_cash_freed():
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = pd.DataFrame({"A": [100, 100, 100, np.nan, np.nan, np.nan], "B": [100.0] * 6}, index=dates)
    panel = make_panel(close)
    panel.fields["returns"].loc[dates[3], "A"] = -0.4  # as Warehouse._apply_delisting_returns would
    targets = pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates)
    targets.loc[dates[3]:, "A"] = 0.0
    out = simulate(targets, panel, SimConfig(**FREE), "daily")
    assert out["returns"].iloc[3] == pytest.approx(-0.2)
    assert (out["weights"]["A"].iloc[3:] == 0).all()


def test_drawdown_breaker_flattens_then_reenters():
    base = np.array([0.0, -0.06, -0.06, -0.06, -0.06, 0.01, 0.01, 0.01, 0.01])
    cfg = SimConfig(dd_limit=0.10, dd_cooldown_days=3, **FREE)
    net, scale, active, _ = _risk_overlay(base, np.ones(len(base)), cfg)
    # -6%, -11.6% -> trips at the close of day 2; days 3-5 are flat; day 6 is back in
    assert scale.tolist() == pytest.approx([1, 1, 1, 0, 0, 0, 1, 1, 1])
    assert net[3] == 0 and net[6] == pytest.approx(0.01)
    assert active.tolist() == [False, False, True, True, True, False, False, False, False]


def test_vol_target_hits_target_without_lookahead():
    rng = np.random.default_rng(5)
    base = rng.normal(0.0003, 0.02, 3000)  # ~32% vol book
    cfg = SimConfig(vol_target=0.10, vol_lookback=60, max_leverage=3, **FREE)
    net, scale, _, _ = _risk_overlay(base, np.ones(3000), cfg)
    assert net[100:].std() * np.sqrt(252) == pytest.approx(0.10, rel=0.1)
    # the multiplier for day t is known at t-1: changing day t's return cannot change it
    bumped = base.copy()
    bumped[2000] = 0.5
    _, scale2, _, _ = _risk_overlay(bumped, np.ones(3000), cfg)
    assert np.array_equal(scale[:2001], scale2[:2001])


def test_vol_target_does_not_lever_an_uninvested_book():
    base = np.r_[np.zeros(200), np.random.default_rng(0).normal(0, 0.01, 50)]
    _, scale, _, _ = _risk_overlay(base, np.ones(250), SimConfig(vol_target=0.1, max_leverage=5, **FREE))
    assert scale[:215].max() == 1.0


def test_rebalance_flags_mark_period_ends():
    dates = pd.bdate_range("2024-01-01", "2024-02-29")
    weekly = dates[rebalance_flags(dates, "weekly")]
    assert (weekly.dayofweek == 4).all()
    monthly = dates[rebalance_flags(dates, "monthly")]
    assert [d.strftime("%m-%d") for d in monthly] == ["01-31", "02-29"]
    assert not rebalance_flags(dates[:-1], "monthly")[-1]  # 02-28 is not the month's last session


def test_backtest_has_no_lookahead(panel, momentum_spec):
    cut = panel.dates[700]
    full = run_backtest(momentum_spec, panel).returns.loc[:cut]
    truncated = run_backtest(momentum_spec, panel.slice(None, cut)).returns
    # the truncated run's last session may differ only through the period-end flag
    pd.testing.assert_series_equal(full.iloc[:-2], truncated.iloc[:-2])


def test_backtest_reports_the_minimum_metric_set(panel, momentum_spec):
    result = run_backtest(momentum_spec, panel, SimConfig(vol_target=0.1, dd_limit=0.2))
    required = {"sharpe", "max_drawdown", "cumulative_return", "annualised_return", "ic_mean", "information_ratio", "deflated_sharpe"}
    assert required <= set(result.metrics)
    # gross is 1.0 at each rebalance and only drifts with prices in between
    assert result.weights.abs().sum(axis=1).max() < 1.25
    assert result.exposure_scale.max() <= result.config.max_leverage


def test_quantile_weights_are_dollar_neutral_and_capped(panel, momentum_spec):
    from oaf.signal import target_weights

    w = target_weights(momentum_spec, panel).dropna(how="all")
    assert np.allclose(w.sum(axis=1), 0, atol=1e-9)
    assert np.allclose(w.abs().sum(axis=1), 1.0, atol=1e-9)
    assert w.abs().max().max() <= 0.2 + 1e-12
    assert (w[~panel.mask.loc[w.index]].fillna(0) == 0).all().all()  # never hold a non-member
