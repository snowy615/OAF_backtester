import numpy as np
import pytest

from oaf.signal import validate_spec
from oaf.sim import SimConfig
from oaf.spec import StrategySpec
from oaf.sweep import apply_overrides, expand_grid, parse_grid, run_sweep, walk_forward


def test_grid_parsing_and_expansion():
    grid = parse_grid(["lookback=20,60", "rebalance=weekly,monthly", "sim.vol_target=none,0.1"])
    assert grid == {"lookback": [20.0, 60.0], "rebalance": ["weekly", "monthly"], "sim.vol_target": [None, 0.1]}
    assert len(expand_grid(grid)) == 8
    assert len(expand_grid(grid, max_trials=3)) == 3


def test_overrides_route_to_spec_and_sim(momentum_spec):
    spec, cfg = apply_overrides(momentum_spec, SimConfig(), {"lookback": 33.4, "rebalance": "monthly", "sim.dd_limit": 0.2})
    assert spec.param_values()["lookback"] == 33 and spec.rebalance == "monthly" and cfg.dd_limit == 0.2
    assert momentum_spec.param_values()["lookback"] == 60  # original untouched
    with pytest.raises(KeyError):
        apply_overrides(momentum_spec, SimConfig(), {"nope": 1})
    with pytest.raises(KeyError):
        apply_overrides(momentum_spec, SimConfig(), {"sim.nope": 1})


def test_sweep_ranks_trials_and_deflates_the_winner(panel, momentum_spec):
    sweep = run_sweep(momentum_spec, panel, {"lookback": [20, 60, 120], "q": [0.1, 0.3]})
    assert len(sweep.table) == 6 and sweep.returns.shape[1] == 6
    assert sweep.table["sharpe"].is_monotonic_decreasing
    best = sweep.best.metrics
    assert best["n_trials"] == 6 and best["sharpe"] == pytest.approx(sweep.table["sharpe"].iloc[0])
    assert best["deflated_sharpe"] < best["psr"]
    assert set(sweep.best_overrides) == {"lookback", "q"}


def test_walk_forward_is_out_of_sample(panel, momentum_spec):
    wf = walk_forward(momentum_spec, panel, {"lookback": [20, 60, 120]}, train_days=300, test_days=120)
    folds = wf.folds
    assert len(folds) >= 3
    assert (folds["train_start"] < folds["test_start"]).all() and (folds["test_start"] <= folds["test_end"]).all()
    assert (folds["test_start"].iloc[1:].values > folds["test_end"].iloc[:-1].values).all()
    assert wf.oos_returns.index.is_unique and wf.oos_returns.index.is_monotonic_increasing
    assert "deflated_sharpe" not in wf.metrics and np.isfinite(wf.metrics["sharpe"])
    with pytest.raises(ValueError, match="not enough history"):
        walk_forward(momentum_spec, panel, {"lookback": [20]}, train_days=5000)


def test_spec_roundtrip_and_validation(tmp_path, momentum_spec, panel):
    path = momentum_spec.save(tmp_path / "s.json")
    assert StrategySpec.load(path) == momentum_spec
    assert validate_spec(momentum_spec, panel.field_names()) == []
    bad = momentum_spec.model_copy(update={"signal": "rank(ts_mean(earnings, lookback))"})
    problems = validate_spec(bad, panel.field_names())
    assert any("earnings" in p for p in problems) and any("skip" in p and "never used" in p for p in problems)


def test_example_specs_are_valid_and_run(panel):
    from pathlib import Path

    from oaf.sim import run_backtest

    for path in sorted(Path(__file__).parent.parent.glob("examples/strategies/*.json")):
        spec = StrategySpec.load(path)
        assert validate_spec(spec, panel.field_names()) == [], path.name
        result = run_backtest(spec, panel)
        assert result.turnover.sum() > 0, path.name
