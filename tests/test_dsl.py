import numpy as np
import pandas as pd
import pytest

from oaf import dsl
from oaf.dsl import DSLError


def test_arithmetic_and_params(toy_panel):
    out = dsl.evaluate("close * k + 1", toy_panel, {"k": 2.0})
    pd.testing.assert_frame_equal(out, toy_panel.fields["close"] * 2 + 1)


def test_ts_ops_match_pandas(toy_panel):
    close = toy_panel.fields["close"]
    pd.testing.assert_frame_equal(dsl.evaluate("ts_mean(close, n)", toy_panel, {"n": 5}), close.rolling(5).mean())
    pd.testing.assert_frame_equal(dsl.evaluate("ts_return(close, 3)", toy_panel, {}), close / close.shift(3) - 1)


def test_rank_is_cross_sectional_and_respects_universe(toy_panel):
    toy_panel.mask.iloc[:, 0] = False  # A is not a member
    try:
        out = dsl.evaluate("rank(close)", toy_panel, {})
        assert out["A"].isna().all()
        assert out.iloc[-1].max() == 1.0 and out.iloc[-1].notna().sum() == 5
    finally:
        toy_panel.mask.iloc[:, 0] = True


def test_conditions_and_ifexp(toy_panel):
    cond = dsl.evaluate_condition("close > ts_mean(close, 5) and not (close > 1e9)", toy_panel, {})
    assert cond.dtypes.eq(bool).all() and cond.any().any()
    out = dsl.evaluate("1 if close > ts_mean(close, 5) else -1", toy_panel, {})
    assert set(np.unique(out.values)) <= {1.0, -1.0}


def test_scalar_expressions():
    assert dsl.evaluate_scalar("q / 2", {"q": 0.4}) == pytest.approx(0.2)
    with pytest.raises(DSLError):
        dsl.evaluate_scalar("close", {})


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "close.values",
        "close[0]",
        "(lambda: 1)()",
        "[c for c in close]",
        "open('x')",
        "ts_mean(close, n=5)",
        "ts_mean(close)",
        "nope(close)",
        "unknown_field + 1",
    ],
)
def test_unsafe_or_invalid_expressions_are_rejected(toy_panel, expr):
    with pytest.raises(DSLError):
        dsl.evaluate(expr, toy_panel, {})


@pytest.mark.parametrize("expr", ["ts_delay(close, -1)", "ts_delay(close, 0)", "ts_mean(close, -5)", "ts_delta(close, n)"])
def test_lookahead_windows_are_rejected(toy_panel, expr):
    with pytest.raises(DSLError):
        dsl.evaluate(expr, toy_panel, {"n": -2})


def test_no_lookahead_truncation_invariance(panel):
    """A signal's value on day t must not change when later data is removed."""
    expr = "rank(ts_zscore(close, 20)) - rank(ts_corr(returns, volume, 30)) + zscore(rsi(close, 14)) + ema(returns, 10)"
    full = dsl.evaluate(expr, panel, {})
    cut = panel.dates[600]
    truncated = dsl.evaluate(expr, panel.slice(None, cut), {})
    pd.testing.assert_frame_equal(full.loc[:cut], truncated)


def test_group_neutralize_removes_sector_means(panel):
    out = dsl.evaluate('group_neutralize(ts_return(close, 20), "sector")', panel, {})
    row = out.iloc[-1].dropna()
    sector_means = row.groupby(panel.groups["sector"].reindex(row.index)).mean()
    assert np.allclose(sector_means, 0, atol=1e-12)


def test_names_in():
    assert dsl.names_in("rank(ts_mean(close, n)) > thr") == {"close", "n", "thr"}
