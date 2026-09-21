import pandas as pd
import pytest

from oaf.deploy.ibkr import DeploymentBlocked, IBKRConfig, check_gates, execute_plan
from oaf.deploy.orders import build_plan, diff_orders, target_shares
from oaf.sim import SimConfig


def test_target_shares_round_towards_zero():
    shares = target_shares(pd.Series({"A": 0.5, "B": -0.25, "C": 0.0001}), pd.Series({"A": 30.0, "B": 7.0, "C": 500.0}), 1000)
    assert shares.to_dict() == {"A": 16, "B": -35}


def test_diff_orders_sells_first_and_skips_dust():
    prices = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0})
    orders = diff_orders({"A": 100, "B": 50, "D": 10}, {"A": 40, "C": 30, "D": 11}, prices, min_notional=50)
    assert [(o.side, o.ticker, o.quantity) for o in orders] == [("SELL", "A", 60), ("SELL", "B", 50), ("BUY", "C", 30)]


def test_build_plan_sizes_to_capital(panel, momentum_spec):
    plan = build_plan(momentum_spec, panel, capital=250_000, current={"SYN000": 10, "AAPL": 5})
    prices = panel.fields["raw_close"].ffill().iloc[-1]
    gross = sum(abs(n) * prices[t] for t, n in plan.target_shares.items())
    assert 0.9 * 250_000 < gross <= 250_000
    assert any("AAPL" in n for n in plan.notes) and "AAPL" not in {o.ticker for o in plan.orders}
    assert plan.mode == "paper" and plan.decision_date <= plan.as_of


def test_breaker_flattens_the_deployment(panel, momentum_spec):
    plan = build_plan(momentum_spec, panel, 100_000, SimConfig(dd_limit=0.0001, dd_cooldown_days=10_000))
    assert plan.exposure_scale == 0 and plan.target_shares == {}


def test_live_trading_is_locked(monkeypatch, panel, momentum_spec):
    monkeypatch.delenv("OAF_ENABLE_LIVE_TRADING", raising=False)
    with pytest.raises(DeploymentBlocked, match="locked"):
        check_gates(IBKRConfig(mode="live", allow_live=True))
    monkeypatch.setenv("OAF_ENABLE_LIVE_TRADING", "YES-I-UNDERSTAND")
    with pytest.raises(DeploymentBlocked):
        check_gates(IBKRConfig(mode="live", allow_live=False))
    check_gates(IBKRConfig(mode="live", allow_live=True, port=7496))
    with pytest.raises(DeploymentBlocked, match="live-trading port"):
        check_gates(IBKRConfig(mode="paper", port=7496))

    plan = build_plan(momentum_spec, panel, 100_000)
    with pytest.raises(DeploymentBlocked, match="built for paper"):
        execute_plan(plan, IBKRConfig(mode="live", allow_live=True, port=7496))
    lines = execute_plan(plan, IBKRConfig(mode="paper", port=7497))
    assert lines and all(line.startswith("DRY RUN") for line in lines)
