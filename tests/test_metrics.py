import numpy as np
import pandas as pd
import pytest

from oaf import metrics as M


def test_drawdown_and_returns_exact():
    r = pd.Series([0.10, -0.50, 0.20, 1.00])
    assert M.cumulative_return(r) == pytest.approx(1.1 * 0.5 * 1.2 * 2 - 1)
    assert M.max_drawdown(r) == pytest.approx(-0.5)
    assert M.max_drawdown(pd.Series([-0.1, 0.0])) == pytest.approx(-0.1)  # loss from the starting capital counts


def test_sharpe_and_annualised_return():
    r = pd.Series([0.01, -0.01] * 126 + [0.01])
    assert M.sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert M.annualised_return(pd.Series([0.001] * 252)) == pytest.approx(1.001**252 - 1)


def test_information_ratio_zero_for_benchmark_hugger():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.01, 500))
    assert np.isnan(M.information_ratio(b, b))
    assert M.information_ratio(b + 0.001 + rng.normal(0, 0.001, 500), b) > 5


def test_ic_detects_a_perfect_and_a_useless_signal():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2022-01-03", periods=300)
    rets = pd.DataFrame(rng.normal(0, 0.02, (300, 30)), index=dates)
    oracle = M.forward_returns(rets, 5)
    assert M.information_coefficient(oracle, rets, 5)["ic_mean"] == pytest.approx(1.0)
    noise = pd.DataFrame(rng.normal(size=(300, 30)), index=dates)
    assert abs(M.information_coefficient(noise, rets, 5)["ic_mean"]) < 0.05


def test_forward_returns_start_the_day_after():
    rets = pd.DataFrame({"A": [0.5, 0.10, 0.20, 0.0]})
    fwd = M.forward_returns(rets, 2)
    assert fwd["A"].iloc[0] == pytest.approx(1.1 * 1.2 - 1)  # excludes day 0's own return


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0006, 0.01, 1500))
    psr = M.probabilistic_sharpe(r)
    d10 = M.deflated_sharpe(r, 10, 0.0005)
    d1000 = M.deflated_sharpe(r, 1000, 0.0005)
    assert M.deflated_sharpe(r, 1, 0.0) == pytest.approx(psr)
    assert psr > d10 > d1000
    assert M.expected_max_sharpe(1000, 0.0005) > M.expected_max_sharpe(10, 0.0005) > 0
