# OAF Backtester

Natural-language-to-backtest platform for the **Oxford Alpha Fund**.

Pitch a strategy in plain English → Claude structures it into a rules-based signal with
named, tunable parameters → it is traded on an internal simulated portfolio → parameters are
swept and walk-forward validated → performance is reported → the validated strategy is sized
to a capital allocation and traded on Interactive Brokers (paper in Year 1, live in Year 2).

```
 idea (str) ──Claude──▶ StrategySpec ──▶ scores ──▶ target weights ──▶ simulated book ──▶ metrics
                            ▲                                              │
                            └────────── sweep / walk-forward / refine ◀────┘
                                                                           ▼
                                                        DeploymentPlan ──▶ IBKR (paper → live)
```

Every stage hands a clean, serialisable object to the next, so you can loop back and re-tune
at any point without rebuilding anything. The product spec is in [docs/PRD.md](docs/PRD.md).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or pick extras: llm, ibkr, yahoo, dev
pytest                           # 75 tests, ~3s
cp .env.example .env             # then set MASSIVE_API_KEY (and ANTHROPIC_API_KEY for `oaf pitch`)
```

Load real market data from **Massive** (massive.com, formerly Polygon.io) into the default warehouse:

```bash
# a list of names (plus every split, and delisting dates from the reference list)
oaf fetch-massive --tickers AAPL MSFT NVDA JPM XOM --start 2016-01-01 --universe mega_caps

# the whole US common-stock market, one grouped-daily call per session, with financials
oaf fetch-massive --market --start 2016-01-01 --fundamentals

# an index constituent list (dated snapshots or start/end intervals) as a point-in-time universe
oaf fetch-massive --tickers-file data/sp500_history.csv --universe sp500 --start 2016-01-01

oaf fetch-massive                # later: refresh from the last stored date
oaf info                         # what the warehouse holds, and whether it looks survivorship-biased
```

Then run the workflow:

```bash
oaf backtest    examples/strategies/momentum_12_1.json --open
oaf sweep       examples/strategies/momentum_12_1.json --grid lookback=60,120,250 skip=1,5,21
oaf walkforward examples/strategies/momentum_12_1.json --train-days 504 --test-days 252
oaf dashboard --open             # runs/index.html: every run side by side
oaf deploy      examples/strategies/momentum_12_1.json --capital 100000   # dry run
```

With a Claude key:

```bash
oaf pitch "Buy stocks with the highest earnings yield that are also in an uptrend; rebalance monthly"
oaf backtest strategies/<name>.json --open
oaf pitch "make it sector-neutral and long-only" --refine strategies/<name>.json
```

No key handy? `oaf demo-data -w data/demo` writes a synthetic market (with delistings, index
changes, splits and lagged fundamentals) and every command takes `-w data/demo`.

## The workflow

| # | Stage | Command | Module |
|---|---|---|---|
| 1 | Pitch a strategy in plain English | `oaf pitch "<idea>"` | `oaf.llm.idea_to_signal` |
| 2 | Claude structures it into a signal | ↑ produces `strategies/<name>.json` | `oaf.spec`, `oaf.dsl` |
| 3 | Entry/exit conditions + frequency, all adjustable | edit the JSON, `--set knob=value`, or `oaf pitch --refine` | `oaf.spec` |
| 4 | Simulate a portfolio trading it | `oaf backtest` | `oaf.signal`, `oaf.sim` |
| 5 | Systematic parameter tuning | `oaf sweep`, `oaf walkforward` | `oaf.sweep` |
| 6 | Performance metrics | printed + `runs/<name>/report.html` + CSVs | `oaf.metrics`, `oaf.report` |
| 7–9 | Connect to IBKR, allocate size, trade | `oaf deploy --capital N [--send]` | `oaf.deploy` |

Data gets in through `oaf fetch-massive` (primary), `oaf universe` for constituent lists,
`oaf ingest <file> [--llm]` for arbitrary files, or `oaf fetch-yahoo` for quick prototypes.

## The strategy object

A `StrategySpec` is plain JSON. Every tunable number is a named entry in `params`; the signal,
the entry/exit rules and the portfolio-construction knobs refer to params by name, so a sweep
only ever overrides `params` (plus `rebalance` and `sim.*` settings).

```json
{
  "name": "momentum_12_1",
  "signal": "rank(ts_delay(close, skip) / ts_delay(close, skip + lookback) - 1)",
  "params": [
    {"name": "lookback", "value": 120, "low": 60, "high": 250, "kind": "int", "description": "..."},
    {"name": "skip", "value": 5, "low": 1, "high": 21, "kind": "int", "description": "..."},
    {"name": "top_quantile", "value": 0.2, "low": 0.1, "high": 0.3, "kind": "float", "description": "..."}
  ],
  "construction": {"mode": "quantile", "direction": "long_short", "weighting": "equal",
                   "long_quantile": "top_quantile", "short_quantile": "top_quantile", "max_weight": "0.1"},
  "rebalance": "weekly",
  "universe": {"name": "all", "min_adv": 5000000}
}
```

Two construction modes cover most pitches:

- **`quantile`** – cross-sectional: long the top fraction by score, short the bottom.
- **`rules`** – per-stock state machine: open on `entry_long` / `entry_short`, hold until
  `exit_long` / `exit_short` (see [rsi_dip_buyer.json](examples/strategies/rsi_dip_buyer.json)).

### Signal DSL

Expressions are parsed with `ast` and evaluated through a whitelist of operators – no `eval`,
no attribute access, nothing that can touch the filesystem. Every operator is backward-looking,
non-positive windows are rejected, and a test asserts that truncating the data never changes
earlier signal values, so **a signal cannot look ahead**.

Time-series (per stock): `ts_mean ts_std ts_sum ts_min ts_max ts_median ts_rank ts_zscore
ts_delay ts_delta ts_return ts_corr ts_skew ts_decay_linear ema rsi` ·
Cross-sectional (across the point-in-time universe): `rank zscore demean scale winsorize
group_neutralize` · Element-wise: `abs sign log sqrt max min clip where is_nan nan_to`, arithmetic,
comparisons, `and/or/not`, `a if cond else b`.

Fields: `open high low close volume returns vwap dollar_volume adv20` plus any fundamentals in
the warehouse. The same operator reference is injected into Claude's system prompt, so the
model and the engine can never drift apart.

## Claude's role (and its limits)

- **Idea → signal** (`oaf pitch`): Claude (`claude-opus-5`, adaptive thinking, structured
  outputs) returns a `StrategySpec`, including a `hypothesis` and an explicit list of
  `assumptions` it made where the pitch was vague. The platform then validates the draft
  against the real DSL and the fields actually in the warehouse; any problems are sent back
  for repair (up to two rounds). What comes out always runs.
- **Data ingestion** (`oaf ingest --llm`): Claude sees a file's header and first 15 rows and
  returns a `ColumnMapping` (layout, column roles, date format, pence-vs-pounds scaling,
  things a human should check). It never touches the numbers – transformation and cleaning
  are deterministic code, and the mapping is saved beside the data for review.
- Server-side refusal fallbacks are enabled by default (`OAF_CLAUDE_FALLBACKS=0` to turn
  off); override the model with `OAF_CLAUDE_MODEL`.

## Data

### Massive (massive.com)

`oaf fetch-massive` is the production data path. It pulls, via the REST API with only the
standard library:

| Massive endpoint | Warehouse table | Notes |
|---|---|---|
| `/v2/aggs/ticker/{t}/range/1/day/…` (per ticker) or `/v2/aggs/grouped/locale/us/market/stocks/{date}` (`--market`) | `prices` | Unadjusted OHLCV stored as-is |
| `/stocks/v1/splits` | `prices.adj_close` | Adjusted close is rebuilt from the splits table so raw and adjusted series are consistent |
| `/v3/reference/tickers`, `active=true` **and** `false` | `meta` | The inactive list is what keeps delisted names in the universe (`delisted_utc` → `delisted_date`) |
| `/v3/reference/tickers/{t}` (`--sectors`) | `meta.sector` | SIC description, one call per ticker |
| `/stocks/financials/v1/income-statements`, `/balance-sheets` (`--fundamentals`) | `fundamentals` | `filing_date` becomes `available_date`; fields `revenue net_income eps eps_diluted shares_outstanding total_equity total_assets total_liabilities cash` |

A membership interval (first bar → delisting date) is written for every fetched ticker under
`--universe NAME`; with `--tickers-file` the file's own dates are used instead. Re-running
without `--start` appends from the last stored date and re-bases stored history for any split
that happened since. Set `MASSIVE_API_KEY` in `.env`; on the free plan pass
`--calls-per-minute 5`. Financials need the Stocks Advanced plan or the Financials add-on.

Massive does not publish index constituents. For a point-in-time S&P 500 (or any index),
supply a CSV of dated snapshots (`date,ticker`) or intervals (`ticker,start_date,end_date`)
via `oaf universe NAME file.csv` or `--tickers-file`. A bare ticker list is accepted but
flagged as survivors-only.

### Point-in-time and survivorship

The warehouse is four parquet tables ([schema](src/oaf/data/schema.py)):

- **prices** – delisted tickers stay; their rows simply stop. Returns use `adj_close`; the raw
  close is kept for order sizing. Halts are forward-filled but never past the final print.
- **membership** – universe constituents as date intervals. `rank()` and friends only see stocks
  that were members *on that day*; leaving the universe closes the position. A spec can also set
  `universe.min_adv` to drop names below a dollar-volume floor, computed point-in-time.
- **fundamentals** – stored with both `period_end` and `available_date`; panels are built on
  `available_date` only, with restatements applying from their own publication date.
- **meta** – sector plus delisting date/return. The delisting return is booked on the first
  session after the last print (−30% assumed if a delisted name has none recorded) and the
  proceeds move to cash.

Cleaning drops certain errors (unparseable rows, non-positive prices, weekend rows, duplicates,
one-day spikes that fully reverse) and *flags* suspicious ones (possible unadjusted splits,
high < low). `oaf info` warns when a multi-year dataset contains no delistings or has no
membership history – the signature of a survivors-only dataset such as a Yahoo download of
today's index.

## The simulated portfolio

- **Timing**: decide at the close of day *d*, fill at the close of *d + execution_lag*
  (default 1), earn from the next session. Weights drift with prices between rebalances.
- **Costs**: commission + slippage per unit of turnover, borrow on short notional.
- **Risk overlay (portfolio-sim layer, not the signal)**: volatility targeting
  (`sim.vol_target`, capped by `sim.max_leverage`) and a drawdown circuit breaker
  (`sim.dd_limit`: flat for `sim.dd_cooldown_days`, high-water mark resets on re-entry).
  Both use only realised returns and pay costs for resizing. They apply to every strategy and
  are sweepable like any other knob: `--set sim.vol_target=0.1 sim.dd_limit=0.15`.

## Metrics and honest tuning

Reported for every run: **Sharpe, max drawdown, cumulative return, annualised return,
Information Coefficient** (daily Spearman rank IC vs forward returns, with a non-overlapping
t-stat) and **Information Ratio** (vs the equal-weight point-in-time universe), plus vol,
Sortino, Calmar, hit rate, turnover.

- `oaf sweep` picks a best configuration in sample, so it reports the **deflated Sharpe
  ratio** (Bailey & López de Prado): the probability the Sharpe is real after allowing for the
  number of configurations tried and the skew/kurtosis of returns.
- `oaf walkforward` is the preferred test: on each training window pick the best
  configuration, score it only on the following unseen window, stitch the out-of-sample
  pieces together, and report how much the score degrades out of sample.

## Dashboard

Every run writes `runs/<name>/report.html`: a single self-contained page (inline CSS and SVG,
no JavaScript, no CDN) with KPI tiles, growth vs the equal-weight universe, drawdown, rolling
Sharpe, a monthly-returns heatmap, exposure, the parameter table, and – for sweeps and
walk-forwards – the configuration table and per-fold results. `oaf dashboard` builds
`runs/index.html` comparing every run with sparklines; `--open` on any command opens the page
in a browser. Reports work offline and can be attached to a pitch as-is. The CSV/JSON
alongside (`returns.csv`, `metrics.json`, `spec.json`, `sweep.csv`, `folds.csv`) are the same
numbers for anyone who wants to plot their own.

## IBKR deployment

`oaf deploy spec.json --capital 100000` sizes the strategy's latest rebalance targets (times
the risk overlay's current multiplier – zero if the circuit breaker is active) to whole
shares, diffs them against current positions, saves `plan.json`, and prints the orders.
Nothing is sent without `--send`. Requires TWS or IB Gateway with the API enabled and
`pip install -e ".[ibkr]"`.

Year 1 is paper-only and the code enforces it: paper mode refuses the live ports (7496/4001)
and any account id that is not an IBKR paper account (`D…`); live mode is locked unless
`OAF_ENABLE_LIVE_TRADING=YES-I-UNDERSTAND` is set **and** `--live` is passed. Orders default
to market-on-close to match the simulator's fill assumption.

## Python API

```python
from oaf import SimConfig, StrategySpec, run_backtest
from oaf.data import Warehouse
from oaf.sweep import run_sweep, walk_forward

panel = Warehouse("data/demo").load_panel("demo_index")
spec = StrategySpec.load("examples/strategies/momentum_12_1.json")

result = run_backtest(spec.with_params(lookback=90), panel, SimConfig(vol_target=0.10, dd_limit=0.15))
print(result.metrics["sharpe"], result.metrics["max_drawdown"])

wf = walk_forward(spec, panel, {"lookback": [60, 120, 250], "skip": [1, 5, 21]}, train_days=504)
print(wf.metrics["sharpe"], wf.degradation)
```

## Status and known limits

- Daily bars, equities, close-to-close fills; no intraday, futures or options yet.
- Costs are linear in turnover (no market-impact model); borrow is a flat rate.
- The Massive client is written against the current REST docs and tested with a scripted
  transport; run one `fetch-massive` with your key to confirm entitlements (history depth and
  financials depend on the plan).
- The IBKR layer is written against `ib_async` and its safety gates and order maths are unit
  tested, but it has not yet been exercised against a running TWS/Gateway – do a `--send`
  shakedown on the paper account before relying on it.
- The Claude stages are tested against a scripted stand-in for the SDK client; run
  `oaf pitch` once with a real key to confirm end to end.
- Index membership history is not available from Massive; without a constituent file the
  universe is "everything listed", filtered by liquidity.
