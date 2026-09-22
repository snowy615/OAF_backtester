"""``oaf`` command line: one subcommand per stage of the pitch -> trade workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from .data.warehouse import Warehouse
from .report import format_metrics, save_run
from .signal import validate_spec
from .sim import SimConfig, run_backtest
from .spec import StrategySpec
from .sweep import apply_overrides, parse_grid, run_sweep, walk_forward


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _panel(args, spec: StrategySpec | None = None):
    wh = Warehouse(args.warehouse)
    universe = args.universe or (spec.universe.name if spec else "all")
    min_dv = spec.universe.min_adv if spec else None
    return wh, wh.load_panel(universe=universe, start=args.start, end=args.end, min_dollar_volume=min_dv)


def _spec_and_config(args) -> tuple[StrategySpec, SimConfig]:
    spec = StrategySpec.load(args.spec)
    overrides = {k: v[0] for k, v in parse_grid(args.set or []).items()}
    return apply_overrides(spec, SimConfig(), overrides)


def _check(spec: StrategySpec, wh: Warehouse, panel) -> None:
    problems = validate_spec(spec, panel.field_names(), wh.groups())
    if problems:
        sys.exit("spec is invalid:\n- " + "\n- ".join(problems))


def _progress(i, n, overrides, result) -> None:
    knobs = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in overrides.items())
    print(f"  [{i}/{n}] {knobs}  sharpe={result.metrics['sharpe']:.2f}", flush=True)


# -- commands ---------------------------------------------------------------------
def cmd_demo_data(args) -> None:
    from .data.synthetic import write_demo_warehouse

    wh = write_demo_warehouse(args.warehouse)
    print(f"wrote synthetic warehouse to {wh.root}")
    print(wh.survivorship_report().summary())


def cmd_info(args) -> None:
    wh = Warehouse(args.warehouse)
    print(f"fields:    {', '.join(wh.available_fields())}")
    print(f"universes: {', '.join(wh.universes())}")
    print(f"groups:    {', '.join(wh.groups()) or 'none'}")
    print(wh.survivorship_report().summary())


def cmd_ingest(args) -> None:
    from .data.ingest import ColumnMapping, ingest_prices, read_table

    mapping = None
    if args.mapping:
        mapping = ColumnMapping.model_validate_json(Path(args.mapping).read_text())
    elif args.llm:
        from .llm.data_mapper import infer_mapping_llm

        mapping = infer_mapping_llm(read_table(args.file), Path(args.file).name)
        print(f"Claude's mapping:\n{mapping.model_dump_json(indent=2)}")
    prices, mapping, report = ingest_prices(args.file, mapping)
    print(report.summary())
    if mapping.notes:
        print(f"notes: {mapping.notes}")
    wh = Warehouse(args.warehouse)
    total = wh.write_prices(prices, source=str(args.file))
    side = wh.root / "mappings" / (Path(args.file).stem + ".json")
    side.parent.mkdir(exist_ok=True)
    side.write_text(mapping.model_dump_json(indent=2) + "\n")
    print(f"warehouse now holds {total} price rows; mapping saved to {side}")
    print(wh.survivorship_report().summary())


def cmd_fetch_yahoo(args) -> None:
    from .data.ingest import clean_prices
    from .data.sources import fetch_yahoo

    prices, report = clean_prices(fetch_yahoo(args.tickers, args.start, args.end))
    print(report.summary())
    wh = Warehouse(args.warehouse)
    print(f"warehouse now holds {wh.write_prices(prices, source='yahoo')} price rows")
    print(wh.survivorship_report().summary())


def _bar(label: str):
    def show(i, n, what):
        print(f"\r  {label} {i}/{n} {what:<24}", end="" if i < n else "\n", flush=True)

    return show


def cmd_fetch_massive(args) -> None:
    from .data import massive as mx
    from .data.ingest import clean_prices

    client = mx.MassiveClient(calls_per_minute=args.calls_per_minute)
    wh = Warehouse(args.warehouse)
    start = args.start
    if start is None:
        existing = wh.read("prices")
        if existing.empty:
            sys.exit("error: --start is required for a fresh warehouse (e.g. --start 2016-01-01)")
        start = (existing["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"refreshing from {start}")

    if args.rebuild_adjusted:  # recompute adj_close from a fresh splits table; no bars are fetched
        splits = mx.fetch_splits(client)
        for f in sorted(wh._path("prices").glob("*.parquet")):
            part = pd.read_parquet(f)
            mx.apply_splits(part, splits).to_parquet(f, index=False)
            print(f"  re-adjusted {f.name}")
        print(f"{client.n_calls} API calls; adj_close rebuilt from {len(splits)} splits")
        return

    print("tickers (active + delisted)...")
    meta = mx.fetch_tickers(client, include_delisted=True)
    universe_name = args.universe

    refreshing = args.start is None
    if args.market:
        exchanges = set(args.exchanges or [])
        listed = meta if not exchanges else meta[meta["primary_exchange"].isin(exchanges)]
        keep_set = set(listed["ticker"])
        splits = mx.fetch_splits(client, start=start if refreshing else None)
        universe_name = universe_name or "us_stocks"
        got_all: set[str] = set()
        workers = 1 if args.calls_per_minute else 8
        for year, chunk in mx.iter_market_years(client, start, args.end, keep=keep_set.__contains__, splits=splits, progress=_bar("day"), workers=workers):
            chunk, report = clean_prices(chunk)
            got_all |= set(chunk["ticker"].unique())
            print(f"  {year}: {report.rows_out} rows written ({', '.join(f'{k} {v}' for k, v in report.dropped.items()) or 'nothing dropped'})")
            wh.write_prices(chunk, source=f"massive:{year}")
        prices = pd.DataFrame({"ticker": sorted(got_all)})  # tickers only; rows are already on disk
        report = None
    else:
        tickers = list(args.tickers or [])
        if args.tickers_file:
            from .data.universes import load_membership

            tickers += load_membership(args.tickers_file, universe_name or "list")["ticker"].unique().tolist()
        tickers = sorted(dict.fromkeys(t.upper() for t in tickers))
        if not tickers:
            sys.exit("error: give --tickers, --tickers-file or --market")
        splits = mx.fetch_splits(client, tickers, start=start if refreshing else None)
        prices = mx.fetch_daily_bars(client, tickers, start, args.end, splits=splits, progress=_bar("ticker"))
        universe_name = universe_name or "list"

    if not args.market:
        prices, report = clean_prices(prices)
        print(report.summary())
        if prices.empty:
            sys.exit("error: Massive returned no bars (check the plan's history limit and the tickers)")
        wh.write_prices(prices, source="massive")
    n = wh.read("prices", columns=["ticker"]).shape[0]
    if refreshing and len(splits):  # a split since the last refresh re-bases the stored history
        _readjust(wh, mx, splits, start)
    got = prices["ticker"].unique()
    meta_rows = meta[meta["ticker"].isin(got)].copy()
    missing = sorted(set(got) - set(meta_rows["ticker"]))
    if missing:
        meta_rows = pd.concat([meta_rows, pd.DataFrame({"ticker": missing})], ignore_index=True)
    if args.sectors:
        meta_rows["sector"] = meta_rows["ticker"].map(mx.fetch_sectors(client, list(meta_rows["ticker"]), progress=_bar("sector")))
    old_meta = wh.read("meta").set_index("ticker") if wh.has("meta") else None
    if old_meta is not None and not args.sectors:  # keep sectors fetched earlier
        meta_rows["sector"] = meta_rows["ticker"].map(old_meta["sector"]).where(lambda s: s.notna(), meta_rows["sector"])
    from .data import schema

    for col in schema.META:
        if col not in meta_rows:
            meta_rows[col] = None
    wh.write_meta(meta_rows[schema.META], source="massive")

    if not args.tickers_file:  # a constituent file is loaded as-is via `oaf universe`
        all_prices = wh.read("prices", columns=["date", "ticker"], filters=[("ticker", "in", list(got))])
        wh.write_membership(mx.membership_from_prices(all_prices, wh.read("meta"), universe_name), source="massive", replace=False)
    else:
        from .data.universes import load_membership

        wh.write_membership(load_membership(args.tickers_file, universe_name), source=str(args.tickers_file))

    if args.fundamentals:
        fund = mx.fetch_fundamentals(client, list(got), start=None, progress=_bar("financials"))
        print(f"  {len(fund)} fundamental observations")
        if len(fund):
            wh.write_fundamentals(fund, source="massive")

    print(f"{client.n_calls} API calls; warehouse now holds {n} price rows, universe '{universe_name}'")
    print(wh.survivorship_report().summary())


def _readjust(wh: Warehouse, mx, splits: pd.DataFrame, since: str) -> None:
    """Re-base stored adj_close for splits since ``since``, one price partition at a time."""
    touched = set(splits.loc[splits["execution_date"] >= pd.Timestamp(since), "ticker"])
    if not touched:
        return
    for f in sorted(wh._path("prices").glob("*.parquet")):
        part = pd.read_parquet(f)
        hit = part["ticker"].isin(touched)
        if hit.any():
            part.loc[hit] = mx.readjust_history(part[hit], splits, since)
            part.to_parquet(f, index=False)


def cmd_universe(args) -> None:
    from .data.universes import load_membership

    mem = load_membership(args.file, args.name)
    wh = Warehouse(args.warehouse)
    wh.write_membership(mem, source=str(args.file), replace=False)
    known = set(wh.read("prices")["ticker"]) if wh.has("prices") else set()
    missing = sorted(set(mem["ticker"]) - known)
    print(f"universe '{args.name}': {mem['ticker'].nunique()} tickers, {len(mem)} membership intervals")
    if missing:
        print(f"  {len(missing)} tickers have no prices yet, e.g. {missing[:8]} - run `oaf fetch-massive --tickers-file {args.file}`")


def cmd_pitch(args) -> None:
    from .llm.idea_to_signal import refine_spec, structure_idea

    wh = Warehouse(args.warehouse)
    fields, universes, groups = wh.available_fields(), wh.universes(), wh.groups()
    if args.refine:
        spec = refine_spec(StrategySpec.load(args.refine), args.idea, fields, universes, groups)
    else:
        spec = structure_idea(args.idea, fields, universes, groups)
    out = spec.save(args.out or Path("strategies") / f"{spec.name}.json")
    print(f"{spec.name}: {spec.description}\n")
    print(f"signal:    {spec.signal}")
    if spec.rules:
        for k, v in spec.rules.model_dump().items():
            if v:
                print(f"{k + ':':<10} {v}")
    print(f"rebalance: {spec.rebalance} | universe: {spec.universe.name}")
    for p in spec.params:
        print(f"  {p.name} = {p.value:g}  [{p.low:g}, {p.high:g}]  {p.description}")
    for a in spec.assumptions:
        print(f"  assumed: {a}")
    print(f"\nsaved to {out}")


def cmd_validate(args) -> None:
    spec = StrategySpec.load(args.spec)
    wh, panel = _panel(args, spec)
    _check(spec, wh, panel)
    print(f"{spec.name}: ok")


def cmd_backtest(args) -> None:
    spec, config = _spec_and_config(args)
    wh, panel = _panel(args, spec)
    _check(spec, wh, panel)
    result = run_backtest(spec, panel, config)
    print(format_metrics(result.metrics, spec.name))
    out = save_run(result, args.out or Path("runs") / spec.name)
    print(f"\nrun saved to {out}/  (report: {out / 'report.html'})")
    _refresh_index(out, args.open)


def cmd_sweep(args) -> None:
    spec, config = _spec_and_config(args)
    wh, panel = _panel(args, spec)
    _check(spec, wh, panel)
    grid = parse_grid(args.grid) if args.grid else spec.default_grid(args.points)
    sweep = run_sweep(spec, panel, grid, config, args.objective, args.max_trials, progress=_progress)
    knobs = list(grid)
    cols = knobs + ["sharpe", "max_drawdown", "annualised_return", "ic_mean", "information_ratio", "annual_turnover"]
    print("\n" + sweep.table[cols].head(args.top).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n" + format_metrics(sweep.best.metrics, f"best of {len(sweep.trials)}: {sweep.best_overrides}"))
    print("\nThis configuration was chosen in sample; trust the deflated Sharpe, and confirm with `oaf walkforward`.")
    out = Path(args.out or Path("runs") / f"{spec.name}_sweep")
    save_run(sweep.best, out, sweep_table=sweep.table)
    sweep.table.to_csv(out / "sweep.csv", index_label="trial")
    print(f"sweep saved to {out}/ (spec.json there is the tuned strategy; report: {out / 'report.html'})")
    _refresh_index(out, args.open)


def cmd_walkforward(args) -> None:
    spec, config = _spec_and_config(args)
    wh, panel = _panel(args, spec)
    _check(spec, wh, panel)
    grid = parse_grid(args.grid) if args.grid else spec.default_grid(args.points)
    wf = walk_forward(
        spec, panel, grid, config, args.train_days, args.test_days, args.expanding, args.objective, args.max_trials,
        progress=_progress,
    )
    print("\n" + wf.folds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + format_metrics(wf.metrics, f"{spec.name}: stitched out-of-sample"))
    print(f"\nmean out-of-sample minus in-sample {wf.objective}: {wf.degradation:+.2f}")
    out = Path(args.out or Path("runs") / f"{spec.name}_walkforward")
    # the report shows the full-sample run of the spec's own defaults alongside the OOS results
    save_run(run_backtest(spec, panel, config), out, walk_forward=wf)
    (out / "metrics.json").write_text(json.dumps({**wf.metrics, "start": str(wf.oos_returns.index[0].date()), "end": str(wf.oos_returns.index[-1].date())}, indent=2, default=str) + "\n")
    wf.folds.to_csv(out / "folds.csv", index=False)
    wf.oos_returns.rename("oos_return").to_csv(out / "oos_returns.csv", index_label="date")
    print(f"saved to {out}/  (report: {out / 'report.html'})")
    _refresh_index(out, args.open)


def _refresh_index(run_dir: Path, open_browser: bool) -> None:
    from .dashboard import build_index, open_in_browser

    build_index(run_dir.parent)
    if open_browser:
        open_in_browser(run_dir / "report.html")


def cmd_dashboard(args) -> None:
    from .dashboard import build_index, open_in_browser

    out = build_index(args.runs)
    print(f"dashboard: {out}")
    if args.open:
        open_in_browser(out)


def cmd_deploy(args) -> None:
    from .deploy.ibkr import DeploymentBlocked, IBKRBroker, IBKRConfig, check_gates, execute_plan
    from .deploy.orders import build_plan

    spec, config = _spec_and_config(args)
    wh, panel = _panel(args, spec)
    _check(spec, wh, panel)
    cfg = IBKRConfig(mode="live" if args.live else "paper", account=args.account, order_type=args.order_type, allow_live=args.live)
    try:
        check_gates(cfg)
        current = {}
        if args.send or args.read_positions:
            with IBKRBroker(cfg) as broker:
                current = broker.positions()
        plan = build_plan(spec, panel, args.capital, config, cfg.mode, current)
        print(plan.summary())
        stale = (pd.Timestamp.today().normalize() - pd.Timestamp(plan.as_of)).days
        if stale > 4:
            print(f"WARNING: data is {stale} days old; refresh the warehouse before trading on it")
        plan.save(Path(args.out or Path("runs") / f"{spec.name}_deploy") / "plan.json")
        print()
        print("\n".join(execute_plan(plan, cfg, send=args.send)))
    except DeploymentBlocked as e:
        sys.exit(f"BLOCKED: {e}")


# -- parser -------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oaf", description="Oxford Alpha Fund backtester")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_, data=True, spec=False):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        if data:
            sp.add_argument("--warehouse", "-w", default="data/warehouse", help="warehouse directory")
        if spec:
            sp.add_argument("spec", help="strategy spec JSON")
            sp.add_argument("--universe", help="override the spec's universe")
            sp.add_argument("--start")
            sp.add_argument("--end")
            sp.add_argument("--set", nargs="*", metavar="KNOB=VALUE", help="override knobs, e.g. lookback=90 sim.vol_target=0.1")
            sp.add_argument("--out", help="output directory")
            sp.add_argument("--open", action="store_true", help="open the HTML report in a browser")
        return sp

    add("demo-data", cmd_demo_data, "write a synthetic warehouse to play with")
    add("info", cmd_info, "show a warehouse's fields, universes and survivorship health")

    sp = add("ingest", cmd_ingest, "normalise + clean a price file into the warehouse")
    sp.add_argument("file")
    sp.add_argument("--llm", action="store_true", help="let Claude work out the file's layout")
    sp.add_argument("--mapping", help="use a saved ColumnMapping JSON")

    sp = add("fetch-yahoo", cmd_fetch_yahoo, "download daily bars from Yahoo (survivors only!)")
    sp.add_argument("tickers", nargs="+")
    sp.add_argument("--start", required=True)
    sp.add_argument("--end")

    sp = add("fetch-massive", cmd_fetch_massive, "load real market data from Massive (massive.com) into the warehouse")
    sp.add_argument("--tickers", nargs="*", help="tickers to fetch")
    sp.add_argument("--tickers-file", help="CSV of tickers (optionally with date / start_date,end_date columns) - also loaded as a universe")
    sp.add_argument("--market", action="store_true", help="every US common stock via the grouped-daily endpoint")
    sp.add_argument("--exchanges", nargs="*", help="with --market: keep only these primary exchanges, e.g. XNYS XNAS")
    sp.add_argument("--start", help="YYYY-MM-DD; omit to refresh from the last stored date")
    sp.add_argument("--end")
    sp.add_argument("--universe", help="membership name to record (default: 'list' or 'us_stocks')")
    sp.add_argument("--fundamentals", action="store_true", help="also load quarterly financials (needs the Financials plan)")
    sp.add_argument("--sectors", action="store_true", help="also fetch SIC sectors (one call per ticker)")
    sp.add_argument("--calls-per-minute", type=int, help="throttle, e.g. 5 on the free plan")
    sp.add_argument("--rebuild-adjusted", action="store_true", help="only recompute adj_close from a fresh splits table")

    sp = add("universe", cmd_universe, "load an index constituent list as a point-in-time universe")
    sp.add_argument("name")
    sp.add_argument("file")

    sp = add("pitch", cmd_pitch, "plain-English idea -> strategy spec (Claude)")
    sp.add_argument("idea", help="the pitch; with --refine, the change you want")
    sp.add_argument("--refine", metavar="SPEC", help="revise this existing spec instead of starting fresh")
    sp.add_argument("--out")

    add("validate", cmd_validate, "check a spec against the warehouse", spec=True)
    add("backtest", cmd_backtest, "simulate a strategy and report metrics", spec=True)

    for name, fn, help_ in (
        ("sweep", cmd_sweep, "grid-sweep the knobs (reports deflated Sharpe)"),
        ("walkforward", cmd_walkforward, "walk-forward validation: tune on each window, score on the next"),
    ):
        sp = add(name, fn, help_, spec=True)
        sp.add_argument("--grid", nargs="*", metavar="KNOB=V1,V2", help="default: each param's range")
        sp.add_argument("--points", type=int, default=3, help="points per param for the default grid")
        sp.add_argument("--max-trials", type=int)
        sp.add_argument("--objective", default="sharpe")
        if name == "sweep":
            sp.add_argument("--top", type=int, default=10)
        else:
            sp.add_argument("--train-days", type=int, default=756)
            sp.add_argument("--test-days", type=int, default=252)
            sp.add_argument("--expanding", action="store_true")

    sp = add("dashboard", cmd_dashboard, "build runs/index.html comparing every run", data=False)
    sp.add_argument("--runs", default="runs")
    sp.add_argument("--open", action="store_true")

    sp = add("deploy", cmd_deploy, "size the strategy to a capital allocation and trade it on IBKR", spec=True)
    sp.add_argument("--capital", type=float, required=True, help="portfolio size allocated to this strategy")
    sp.add_argument("--send", action="store_true", help="actually place the orders (default: dry run)")
    sp.add_argument("--read-positions", action="store_true", help="dry run, but diff against real IBKR positions")
    sp.add_argument("--live", action="store_true", help="Year 2: live account (also needs OAF_ENABLE_LIVE_TRADING)")
    sp.add_argument("--account")
    sp.add_argument("--order-type", choices=["MOC", "MKT"], default="MOC")
    return p


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    for attr in ("universe", "start", "end"):
        if not hasattr(args, attr):
            setattr(args, attr, None)
    try:
        args.fn(args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")
    except Exception as e:  # LLMError and friends: show the message, not a traceback
        if type(e).__name__ in ("LLMError", "DSLError", "MassiveError"):
            sys.exit(f"error: {e}")
        raise


if __name__ == "__main__":
    main()
