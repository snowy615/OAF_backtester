import json

import pandas as pd

from oaf.dashboard import build_index, run_report
from oaf.report import save_run
from oaf.sim import SimConfig, run_backtest
from oaf.sweep import run_sweep, walk_forward


def test_run_report_is_self_contained_html(panel, momentum_spec):
    result = run_backtest(momentum_spec, panel, SimConfig(vol_target=0.1))
    page = run_report(result)
    assert page.startswith("<!doctype html>") and "<script" not in page
    assert "src=" not in page and "href='http" not in page and 'href="http' not in page  # nothing loaded from the network
    for text in ("Growth of 1", "Drawdown", "Monthly returns", "<code>lookback</code>", momentum_spec.signal.replace("&", "&amp;")):
        assert text in page


def test_save_run_and_index(tmp_path, panel, momentum_spec):
    runs = tmp_path / "runs"
    sweep = run_sweep(momentum_spec, panel, {"lookback": [20, 60]})
    save_run(sweep.best, runs / "a_sweep", sweep_table=sweep.table)
    wf = walk_forward(momentum_spec, panel, {"lookback": [20, 60]}, train_days=300, test_days=120)
    save_run(run_backtest(momentum_spec, panel), runs / "b_wf", walk_forward=wf)
    wf.oos_returns.rename("oos_return").to_csv(runs / "b_wf" / "oos_returns.csv", index_label="date")
    (runs / "b_wf" / "metrics.json").write_text(json.dumps(wf.metrics))

    a = (runs / "a_sweep" / "report.html").read_text()
    assert "Sweep · 2 configurations" in a and "Deflated" in a
    b = (runs / "b_wf" / "report.html").read_text()
    assert "stitched out-of-sample" in b and "Folds" in b

    index = build_index(runs)
    page = index.read_text()
    assert "a_sweep/report.html" in page and "b_wf/report.html" in page and page.count("<svg") >= 2
    assert not (runs / "a_sweep" / "equity.png").exists()
