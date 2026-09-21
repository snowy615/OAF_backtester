# Backtester — PRD

## Overview

- Natural-language-to-backtest platform
- Pitch a strategy in plain English → structured rules-based signal → backtest on internal simulated portfolio → performance metrics → deploy to IBKR
- Built inside OAF's existing fund structure
- Reference for flow/UX: horizon.trade

## Core features

- **Claude interface: idea → signal** — turn an unprocessed idea into a rules-based signal with explicit, adjustable parameters (entry/exit conditions, thresholds, lookbacks, rebalance frequency, universe); output is a parameterised strategy object, every knob named and tunable
- **Data ingestion & cleaning (Claude-assisted)** — ingest and clean raw market data; point-in-time correctness and survivorship-bias handling required; Claude assists normalising messy source formats into warehouse schema
- **Internal simulated portfolio** — simulate a portfolio trading the strategy against historical data; support systematic parameter sweeps, not one-offs
- **IBKR deployment** — connect validated strategies to IBKR; Year 1 paper only, Year 2 live capital; allocate a chosen portfolio size to a strategy and trade it

## Workflow

1. Pitch strategy (plain-English idea)
2. Structure into signal (Claude)
3. Model sets entry/exit conditions + frequency (adjustable)
4. Simulate portfolio trading the strategy
5. Systematic parameter tuning — sweep, refine, re-run
6. Performance metrics reported
7. Connect to IBKR
8. Allocate portfolio size to strategy
9. Trade it (paper Y1 → live Y2)

Each stage passes a clean object to the next; can loop back and re-tune without rebuilding.

## Metrics (minimum set)

- Sharpe ratio
- Max drawdown
- Cumulative return
- Annualised return
- Information Coefficient (IC)
- Information Ratio (IR)

Report deflated Sharpe when selecting a best config from many; prefer walk-forward over single in-sample fit.
Vol targeting and drawdown circuit breakers live at the portfolio-sim layer.

## Year 1 vs Year 2

| | Year 1 | Year 2 |
|---|---|---|
| IBKR | Paper portfolio | Live capital |
| Goal | Build platform + paper-trade top strategies | Launch live quant sleeve |
| Pitch cycle | Every termly pitch requires a novel strategy backtested on the platform | Best strategies allocated real size |

## Where each requirement lives in the code

| Requirement | Implementation |
|---|---|
| Idea → parameterised strategy object | `oaf.llm.idea_to_signal`, `oaf.spec.StrategySpec` |
| Every knob named and tunable | `StrategySpec.params`; expressions reference params by name |
| Claude-assisted normalisation | `oaf.llm.data_mapper` → `oaf.data.ingest.ColumnMapping` |
| Point-in-time correctness | `Warehouse.load_panel` (`available_date` fundamentals, membership intervals), backward-looking DSL |
| Survivorship-bias handling | delisted names kept, delisting returns booked, `Warehouse.survivorship_report` |
| Simulated portfolio | `oaf.sim.simulate` |
| Systematic sweeps | `oaf.sweep.run_sweep` |
| Walk-forward preferred | `oaf.sweep.walk_forward` |
| Minimum metric set + deflated Sharpe | `oaf.metrics` |
| Vol targeting + drawdown breaker at the sim layer | `oaf.sim._risk_overlay` (`SimConfig.vol_target`, `dd_limit`) |
| IBKR, allocate size, trade | `oaf.deploy.orders.build_plan`, `oaf.deploy.ibkr` |
| Paper Y1 → live Y2 | `oaf.deploy.ibkr.check_gates` |
| Clean object between stages, loop back freely | `StrategySpec` → `BacktestResult` → `SweepResult` / `WalkForwardResult` → `DeploymentPlan` |
