"""Human-readable output and run persistence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .sim import BacktestResult

_PCT = {"max_drawdown", "cumulative_return", "annualised_return", "annualised_vol", "hit_rate", "ic_hit_rate", "benchmark_annualised_return"}
_HEADLINE = [
    ("sharpe", "Sharpe ratio"),
    ("max_drawdown", "Max drawdown"),
    ("cumulative_return", "Cumulative return"),
    ("annualised_return", "Annualised return"),
    ("ic_mean", "Information Coefficient (mean rank IC)"),
    ("information_ratio", "Information Ratio (vs equal-weight universe)"),
]
_DETAIL = [
    ("annualised_vol", "Annualised vol"),
    ("sortino", "Sortino"),
    ("calmar", "Calmar"),
    ("hit_rate", "Daily hit rate"),
    ("ic_ir", "IC information ratio"),
    ("ic_tstat", "IC t-stat (non-overlapping)"),
    ("benchmark_annualised_return", "Benchmark annualised return"),
    ("annual_turnover", "Annual turnover (x book)"),
    ("psr", "Probabilistic Sharpe, P(SR > 0)"),
    ("deflated_sharpe", "Deflated Sharpe"),
]


def fmt(key: str, value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if key in _PCT:
        return f"{value:+.2%}" if key != "hit_rate" and key != "ic_hit_rate" else f"{value:.1%}"
    if key in ("ic_mean",):
        return f"{value:+.4f}"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def format_metrics(metrics: dict, title: str = "") -> str:
    lines = [title, "=" * len(title)] if title else []
    width = max(len(label) for _, label in _HEADLINE + _DETAIL)
    for group in (_HEADLINE, _DETAIL):
        for key, label in group:
            if key in metrics:
                extra = f"  (best of {metrics['n_trials']} trials)" if key == "deflated_sharpe" and metrics.get("n_trials", 1) > 1 else ""
                lines.append(f"{label:<{width}}  {fmt(key, metrics[key])}{extra}")
        lines.append("")
    if "start" in metrics:
        lines.append(f"Period: {metrics['start']} -> {metrics['end']}")
    return "\n".join(lines).rstrip()


def tearsheet(result: BacktestResult) -> str:
    spec, cfg = result.spec, result.config
    knobs = "\n".join(f"| `{p.name}` | {p.value:g} | {p.low:g} - {p.high:g} | {p.description} |" for p in spec.params)
    overlay = []
    if cfg.vol_target is not None:
        overlay.append(f"vol target {cfg.vol_target:.0%} (lookback {cfg.vol_lookback}d, max leverage {cfg.max_leverage:g}x)")
    if cfg.dd_limit is not None:
        days_off = int(result.breaker_active.sum())
        overlay.append(f"drawdown breaker at -{cfg.dd_limit:.0%} ({cfg.dd_cooldown_days}d cooldown; flat for {days_off} sessions)")
    return f"""# {spec.name}

{spec.description}

**Idea:** {spec.idea or "n/a"}

**Signal:** `{spec.signal}`
**Universe:** {spec.universe.name} | **Rebalance:** {spec.rebalance} | **Construction:** {spec.construction.mode}, {spec.construction.direction}, {spec.construction.weighting}-weighted
**Costs:** {cfg.commission_bps + cfg.slippage_bps:g} bps per unit turnover, {cfg.borrow_bps_annual:g} bps/yr borrow, execution lag {cfg.execution_lag}d
**Risk overlay:** {"; ".join(overlay) or "none"}

| Param | Value | Range | Meaning |
|---|---|---|---|
{knobs}

```
{format_metrics(result.metrics)}
```
"""


def save_run(result: BacktestResult, out_dir: str | Path, sweep_table=None, walk_forward=None) -> Path:
    """Persist everything needed to audit or re-tune the run, plus the HTML report."""
    from .dashboard import run_report

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.spec.save(out / "spec.json")
    (out / "config.json").write_text(json.dumps(result.config.model_dump(), indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2, default=str) + "\n")
    pd.DataFrame(
        {
            "return": result.returns,
            "base_return": result.base_returns,
            "benchmark": result.benchmark,
            "equity": result.equity,
            "turnover": result.turnover,
            "exposure_scale": result.exposure_scale,
        }
    ).to_csv(out / "returns.csv", index_label="date")
    (out / "tearsheet.md").write_text(tearsheet(result))
    (out / "report.html").write_text(run_report(result, sweep_table=sweep_table, walk_forward=walk_forward))
    return out
