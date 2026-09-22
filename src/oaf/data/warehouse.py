"""Parquet-backed warehouse and point-in-time panel construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..panel import Panel
from . import schema

# Shumway (1997): performance-related delistings with a missing delisting return
# average roughly -30%. Used only when meta marks a ticker delisted without a return.
ASSUMED_DELISTING_RETURN = -0.30


@dataclass
class SurvivorshipReport:
    n_tickers: int
    n_delisted: int
    has_membership: bool
    years: float
    warnings: list[str]

    def summary(self) -> str:
        head = (
            f"{self.n_tickers} tickers over {self.years:.1f}y, {self.n_delisted} delisted/ended early, "
            f"membership history: {'yes' if self.has_membership else 'NO'}"
        )
        return "\n".join([head] + [f"  WARNING: {w}" for w in self.warnings])


class Warehouse:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- raw table IO ------------------------------------------------------
    def _path(self, table: str) -> Path:
        return self.root / ("prices" if table == "prices" else f"{table}.parquet")

    def has(self, table: str) -> bool:
        p = self._path(table)
        return p.is_dir() and any(p.glob("*.parquet")) if table == "prices" else p.exists()

    def read(self, table: str, columns: Optional[Sequence[str]] = None, filters=None) -> pd.DataFrame:
        """Read a table. ``filters`` are pyarrow row filters, e.g. ``[("ticker", "in", [...])]``."""
        if not self.has(table):
            return pd.DataFrame(columns=list(columns) if columns else schema.TABLES[table])
        df = pd.read_parquet(self._path(table), columns=list(columns) if columns else None, filters=filters)
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str)
        return df.reset_index(drop=True)

    @staticmethod
    def _canonical(table: str, df: pd.DataFrame) -> pd.DataFrame:
        cols = schema.TABLES[table]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{table}: missing columns {missing}")
        df = df[cols].copy()
        for c in cols:
            if c.endswith("date") or c == "period_end":
                df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
        return df

    def write(self, table: str, df: pd.DataFrame, keys: Sequence[str], source: str = "", replace: bool = False) -> int:
        """Upsert ``df`` into ``table`` on ``keys`` (new rows win). Returns the row count after."""
        if table == "prices":
            return self.write_prices(df, source, replace)
        df = self._canonical(table, df)
        if not replace and self.has(table):
            df = pd.concat([self.read(table), df], ignore_index=True)
        df = df.drop_duplicates(list(keys), keep="last").sort_values(list(keys)).reset_index(drop=True)
        df.to_parquet(self._path(table), index=False)
        self._log(table, len(df), source)
        return len(df)

    def write_prices(self, df: pd.DataFrame, source: str = "", replace: bool = False) -> int:
        """Prices are stored one parquet file per year, so a whole-market history can be
        written and refreshed a year at a time without holding all of it in memory."""
        df = self._canonical("prices", df)
        root = self._path("prices")
        if replace and root.exists():
            for f in root.glob("*.parquet"):
                f.unlink()
        root.mkdir(parents=True, exist_ok=True)
        for year, chunk in df.groupby(df["date"].dt.year):
            f = root / f"year={int(year)}.parquet"
            if f.exists():
                chunk = pd.concat([pd.read_parquet(f), chunk], ignore_index=True)
            chunk = chunk.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"]).reset_index(drop=True)
            chunk.to_parquet(f, index=False)
        total = sum(pq.read_metadata(f).num_rows for f in root.glob("*.parquet"))
        self._log("prices", total, source)
        return total

    def write_membership(self, df: pd.DataFrame, source: str = "", replace: bool = False) -> int:
        return self.write("membership", df, ["universe", "ticker", "start_date"], source, replace)

    def write_fundamentals(self, df: pd.DataFrame, source: str = "", replace: bool = False) -> int:
        return self.write("fundamentals", df, ["ticker", "field", "period_end", "available_date"], source, replace)

    def write_meta(self, df: pd.DataFrame, source: str = "", replace: bool = False) -> int:
        return self.write("meta", df, ["ticker"], source, replace)

    def _log(self, table: str, rows: int, source: str) -> None:
        entry = {"at": datetime.now(timezone.utc).isoformat(), "table": table, "rows": rows, "source": source}
        with open(self.root / "ingest_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    # -- catalogue ---------------------------------------------------------
    def universes(self) -> list[str]:
        named = sorted(self.read("membership")["universe"].unique()) if self.has("membership") else []
        return ["all"] + named

    def fundamental_fields(self) -> list[str]:
        return sorted(self.read("fundamentals")["field"].unique()) if self.has("fundamentals") else []

    def available_fields(self) -> list[str]:
        base = ["open", "high", "low", "close", "volume", "returns", "vwap", "dollar_volume", "adv20"]
        return base + self.fundamental_fields()

    def groups(self) -> list[str]:
        meta = self.read("meta")
        return ["sector"] if len(meta) and meta["sector"].notna().any() else []

    # -- survivorship ------------------------------------------------------
    def survivorship_report(self) -> SurvivorshipReport:
        px = self.read("prices", columns=["date", "ticker"])
        if px.empty:
            return SurvivorshipReport(0, 0, False, 0.0, ["warehouse has no prices"])
        last = px.groupby("ticker")["date"].max()
        end = px["date"].max()
        years = (end - px["date"].min()).days / 365.25
        n_ended = int((last < end - pd.Timedelta(days=10)).sum())
        has_mem = self.has("membership") and len(self.read("membership")) > 0
        warnings = []
        if years >= 3 and n_ended == 0:
            warnings.append(
                "no ticker ever stops trading in a multi-year history - this looks like a survivors-only "
                "dataset (e.g. today's index constituents). Backtests on it will be biased upwards."
            )
        if not has_mem:
            warnings.append("no point-in-time membership table; universe 'all' = whatever has a price that day.")
        return SurvivorshipReport(int(last.size), n_ended, has_mem, years, warnings)

    # -- panel -------------------------------------------------------------
    def load_panel(
        self,
        universe: str = "all",
        start=None,
        end=None,
        tickers: Optional[Sequence[str]] = None,
        fundamentals: Optional[Sequence[str]] = None,
        max_staleness_days: int = 400,
        min_dollar_volume: Optional[float] = None,
        max_gap_sessions: int = 20,
        max_daily_gain: float = 9.0,
    ) -> Panel:
        """Build a point-in-time panel.

        ``fundamentals=None`` loads every fundamental field in the warehouse.
        ``min_dollar_volume`` drops tickers that *never* trade that much in a day - a
        cheap superset of ``Universe.min_adv`` that keeps whole-market panels tractable
        (a 20-day average can only reach the floor if some single day did).
        ``max_gap_sessions`` and ``max_daily_gain`` break a return series where a symbol
        reappears after a long silence or jumps more than 10x in a day: both are data
        artefacts (ticker reuse, missing reverse splits, bad ticks) rather than returns.
        """
        filters = []
        if end is not None:
            filters.append(("date", "<=", pd.Timestamp(end)))
        if tickers:
            filters.append(("ticker", "in", [str(t) for t in tickers]))
        if universe != "all":
            mem = self.read("membership")
            mem = mem[mem["universe"] == universe]
            if mem.empty:
                raise ValueError(f"unknown universe {universe!r}; available: {self.universes()}")
            filters.append(("ticker", "in", sorted(mem["ticker"].unique())))
        if min_dollar_volume is not None:
            dv = self.read("prices", columns=["ticker", "close", "volume"], filters=filters or None)
            dv["dv"] = dv["close"] * dv["volume"]
            liquid = dv.groupby("ticker")["dv"].max()
            keep = sorted(liquid[liquid >= min_dollar_volume].index)
            del dv
            if not keep:
                raise ValueError(f"warehouse {self.root} has no prices for this selection; ingest some first")
            filters.append(("ticker", "in", keep))
        px = self.read("prices", filters=filters or None)
        if px.empty:
            raise ValueError(f"warehouse {self.root} has no prices for this selection; ingest some first")
        # start is applied after indicators' history is available to the caller: keep all
        # rows <= end so lookbacks at `start` are warm, then cut at the very end.

        def piv(col: str) -> pd.DataFrame:
            return px.pivot(index="date", columns="ticker", values=col).sort_index()

        raw_close = piv("close")
        adj = piv("adj_close")
        adj = adj.where(adj.notna(), raw_close)
        factor = (adj / raw_close).replace([np.inf, -np.inf], np.nan)

        # Forward-fill through halts, but never past a ticker's final print.
        alive = adj.bfill().notna()
        px_ff = adj.ffill().where(alive)
        returns = px_ff.pct_change(fill_method=None)
        # A symbol that reappears after a long silence is usually a *different* company
        # (exchanges recycle tickers), so no return is booked across the gap.
        has = adj.notna().values
        pos = np.arange(len(adj))[:, None]
        last_print = pd.DataFrame(np.where(has, pos, np.nan), index=adj.index, columns=adj.columns).ffill().shift(1).values
        rejoin = has & ((pos - last_print) > max_gap_sessions)
        returns = returns.mask(rejoin)
        # A >10x single-session gain is a missing reverse split, a post-bankruptcy re-listing
        # or a bad tick (a $5m/day stock does not really do that), so it is also a break; a
        # >90% crash that such a jump immediately undoes was the bad tick itself.
        jump = returns > max_daily_gain
        tick = (returns < -0.9) & jump.shift(-1, fill_value=False)
        returns = returns.mask(jump | tick)

        meta = self.read("meta")
        returns = _apply_delisting_returns(returns, adj, meta)

        fields: dict[str, pd.DataFrame] = {
            "close": adj,
            "raw_close": raw_close,
            "returns": returns,
            "volume": piv("volume"),
        }
        for col in ("open", "high", "low"):
            fields[col] = piv(col) * factor
        # no intraday data: typical price stands in for VWAP
        hlc = (fields["high"] + fields["low"] + fields["close"]) / 3
        fields["vwap"] = hlc.where(hlc.notna(), adj)
        fields["dollar_volume"] = raw_close * fields["volume"]
        fields["adv20"] = fields["dollar_volume"].rolling(20).mean()

        fund_names = self.fundamental_fields() if fundamentals is None else list(fundamentals)
        if fund_names:
            fund = self.read("fundamentals")
            for name in fund_names:
                fields[name] = _pit_field(fund[fund["field"] == name], adj.index, adj.columns, max_staleness_days)

        mask = raw_close.notna()
        if universe != "all":
            mask &= _membership_mask(mem, adj.index, adj.columns)

        groups = {}
        if len(meta) and meta["sector"].notna().any():
            groups["sector"] = meta.set_index("ticker")["sector"].reindex(adj.columns)

        panel = Panel(fields=fields, mask=mask, groups=groups, universe=universe)
        return panel.slice(start, None) if start is not None else panel


def _membership_mask(mem: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=dates, columns=tickers)
    far = pd.Timestamp.max
    for row in mem.itertuples(index=False):
        if row.ticker in mask.columns:
            stop = far if pd.isna(row.end_date) else row.end_date
            mask.loc[row.start_date:stop, row.ticker] = True
    return mask


def _pit_field(rows: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index, max_staleness_days: int) -> pd.DataFrame:
    """Value known as of each date: keyed on ``available_date``, never ``period_end``."""
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
    if rows.empty:
        return out
    rows = rows.sort_values(["available_date", "period_end"])
    rows = rows.drop_duplicates(["ticker", "available_date"], keep="last")
    vals = rows.pivot(index="available_date", columns="ticker", values="value")
    stamp = vals.notna().mul(vals.index.to_series().astype("int64"), axis=0).where(vals.notna())
    idx = vals.index.union(dates)
    vals = vals.reindex(idx).ffill().reindex(dates)
    stamp = stamp.reindex(idx).ffill().reindex(dates)
    age_days = (pd.Series(dates.astype("int64"), index=dates).values[:, None] - stamp.values) / 8.64e13
    vals = vals.where(age_days <= max_staleness_days)
    return vals.reindex(columns=tickers)


def _apply_delisting_returns(returns: pd.DataFrame, adj: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Book the delisting return on the first session after a delisted ticker's last print."""
    if meta.empty:
        return returns
    returns = returns.copy()
    dates = returns.index
    for row in meta.itertuples(index=False):
        if row.ticker not in returns.columns or pd.isna(row.delisted_date):
            continue
        last = adj[row.ticker].last_valid_index()
        if last is None:
            continue
        pos = dates.get_loc(last) + 1
        if pos < len(dates):
            r = ASSUMED_DELISTING_RETURN if pd.isna(row.delisting_return) else float(row.delisting_return)
            returns.iloc[pos, returns.columns.get_loc(row.ticker)] = r
    return returns
