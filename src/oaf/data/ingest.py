"""Normalise messy price files into the warehouse schema, then clean them.

Two steps, both inspectable:

1. a :class:`ColumnMapping` says how the source's columns map onto the canonical
   schema. It is inferred heuristically, or by Claude (``oaf.llm.data_mapper``)
   for formats the heuristics cannot read, and can be saved/edited as JSON.
2. :func:`clean_prices` applies deterministic cleaning rules and returns a
   :class:`CleaningReport` of everything it dropped or flagged. Claude never
   touches the numbers themselves - only the mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from . import schema


class ColumnMapping(BaseModel):
    """How a source file maps onto the canonical ``prices`` table."""

    layout: Literal["long", "wide"] = "long"
    # long: one row per (date, ticker). wide: one row per date, one column per ticker.
    date_col: str
    ticker_col: Optional[str] = None
    ticker_value: Optional[str] = None  # single-instrument files with no ticker column
    open_col: Optional[str] = None
    high_col: Optional[str] = None
    low_col: Optional[str] = None
    close_col: Optional[str] = None
    adj_close_col: Optional[str] = None
    volume_col: Optional[str] = None
    wide_field: Literal["close", "adj_close"] = "adj_close"  # what the cells of a wide file hold
    date_format: Optional[str] = None  # strptime format; None = let pandas infer
    dayfirst: bool = False
    price_scale: float = 1.0  # e.g. 0.01 for LSE prices quoted in pence
    notes: str = ""


_SYNONYMS = {
    "date_col": ["date", "datetime", "timestamp", "time", "day", "trade_date", "tradedate", "dt", "asof"],
    "ticker_col": ["ticker", "symbol", "sym", "ric", "code", "instrument", "security", "stock", "tic", "bbg", "isin"],
    "open_col": ["open", "o", "px_open", "open_price"],
    "high_col": ["high", "h", "px_high", "high_price"],
    "low_col": ["low", "l", "px_low", "low_price"],
    "close_col": ["close", "c", "px_last", "last", "price", "close_price", "prc", "px_close", "settle"],
    "adj_close_col": ["adj_close", "adjclose", "adj", "adjusted_close", "close_adj", "adj_price", "adjusted"],
    "volume_col": ["volume", "vol", "v", "px_volume", "shares_traded", "qty"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def infer_mapping_heuristic(df: pd.DataFrame) -> ColumnMapping:
    """Best-effort mapping from column names alone. Raises if no date column is found."""
    normed = {_norm(c): c for c in df.columns}
    found: dict[str, Optional[str]] = {}
    for key, names in _SYNONYMS.items():
        found[key] = next((normed[n] for n in names if n in normed), None)
    if found["date_col"] is None:
        raise ValueError(f"could not find a date column among {list(df.columns)}; pass a ColumnMapping or use --llm")

    price_cols = [found[k] for k in ("close_col", "adj_close_col")]
    if not any(price_cols):
        # No recognisable price column: a wide file (date + one column per ticker)?
        others = [c for c in df.columns if c != found["date_col"]]
        numeric = [c for c in others if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
        if found["ticker_col"] is None and len(numeric) >= 2 and len(numeric) == len(others):
            return ColumnMapping(layout="wide", date_col=found["date_col"], notes="inferred wide layout")
        raise ValueError(f"could not find a close/price column among {list(df.columns)}; pass a ColumnMapping or use --llm")
    return ColumnMapping(layout="long", **found)


def apply_mapping(df: pd.DataFrame, m: ColumnMapping, default_ticker: Optional[str] = None) -> pd.DataFrame:
    """Reshape ``df`` to the canonical long ``prices`` columns (no cleaning yet)."""
    dates = pd.to_datetime(df[m.date_col], format=m.date_format, dayfirst=m.dayfirst, errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    dates = dates.dt.normalize()

    if m.layout == "wide":
        body = df.drop(columns=[m.date_col]).apply(pd.to_numeric, errors="coerce") * m.price_scale
        body.insert(0, "date", dates)
        out = body.melt(id_vars="date", var_name="ticker", value_name=m.wide_field)
    else:
        out = pd.DataFrame({"date": dates})
        if m.ticker_col:
            out["ticker"] = df[m.ticker_col].astype(str)
        else:
            ticker = m.ticker_value or default_ticker
            if not ticker:
                raise ValueError("file has no ticker column; set ColumnMapping.ticker_value")
            out["ticker"] = ticker
        for f in schema.PRICE_FIELDS:
            src = getattr(m, f"{f}_col")
            if src is not None:
                col = pd.to_numeric(df[src].astype(str).str.replace(",", ""), errors="coerce")
                out[f] = col if f == "volume" else col * m.price_scale

    for f in schema.PRICE_FIELDS:
        if f not in out.columns:
            out[f] = np.nan
    out["close"] = out["close"].where(out["close"].notna(), out["adj_close"])
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    return out[schema.PRICES]


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    flagged: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def _note(self, bucket: dict[str, int], key: str, rows: pd.DataFrame) -> None:
        if len(rows):
            bucket[key] = bucket.get(key, 0) + len(rows)
            self.examples[key] = [f"{r.ticker} {r.date:%Y-%m-%d}" for r in rows.head(5).itertuples()]

    def summary(self) -> str:
        lines = [f"rows: {self.rows_in} in -> {self.rows_out} out"]
        lines += [f"  dropped {k}: {v} (e.g. {', '.join(self.examples.get(k, []))})" for k, v in self.dropped.items()]
        lines += [f"  FLAGGED {k}: {v} (e.g. {', '.join(self.examples.get(k, []))})" for k, v in self.flagged.items()]
        return "\n".join(lines)


def clean_prices(df: pd.DataFrame, spike: float = 0.5) -> tuple[pd.DataFrame, CleaningReport]:
    """Deterministic cleaning. Drops what is certainly wrong, flags what is only suspicious.

    Dropped: unparseable dates/tickers, missing or non-positive closes, weekend rows,
    duplicate (date, ticker) rows, and one-day bad ticks (a move of more than ``spike``
    that fully reverses the next day). Flagged but kept: jumps that look like unadjusted
    splits, and high/low inconsistencies (which are repaired by swapping).
    """
    rep = CleaningReport(rows_in=len(df))
    df = df.copy()

    bad = df["date"].isna() | df["ticker"].isin(["", "NAN", "NONE"])
    if bad.any():  # no examples: a NaT has no date to print
        rep.dropped["unparseable date/ticker"] = int(bad.sum())
    df = df[~bad]

    bad = df["close"].isna() | (df["close"] <= 0)
    rep._note(rep.dropped, "missing/non-positive close", df[bad])
    df = df[~bad]

    bad = df["date"].dt.dayofweek >= 5
    rep._note(rep.dropped, "weekend date", df[bad])
    df = df[~bad]

    dup = df.duplicated(["date", "ticker"], keep="last")
    rep._note(rep.dropped, "duplicate (date, ticker)", df[dup])
    df = df[~dup].sort_values(["ticker", "date"]).reset_index(drop=True)

    for col in ("open", "high", "low", "adj_close"):
        df.loc[df[col] <= 0, col] = np.nan
    df.loc[df["volume"] < 0, "volume"] = np.nan

    swapped = df["high"] < df["low"]
    rep._note(rep.flagged, "high < low (swapped)", df[swapped])
    df.loc[swapped, ["high", "low"]] = df.loc[swapped, ["low", "high"]].values

    # Bad ticks: a huge move that is undone the very next print.
    px = df["adj_close"].where(df["adj_close"].notna(), df["close"])
    g = px.groupby(df["ticker"])
    prev, nxt = g.shift(1), g.shift(-1)
    r_in, r_round = px / prev - 1, nxt / prev - 1
    tick = (r_in.abs() > spike) & (r_round.abs() < 0.1)
    rep._note(rep.dropped, "bad tick (spike that reverses next day)", df[tick])
    df = df[~tick].reset_index(drop=True)

    # Unadjusted splits: a clean 2:1 / 3:1 / 1:2 style jump in a series with no adj_close.
    raw_r = df["close"] / df.groupby("ticker")["close"].shift(1)
    ratios = np.array([1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2, 3, 4, 5, 10])
    near = np.abs(raw_r.values[:, None] / ratios[None, :] - 1).min(axis=1) < 0.03
    rep._note(rep.flagged, "possible unadjusted split (no adj_close)", df[near & df["adj_close"].isna()])

    rep.dropped = {k: v for k, v in rep.dropped.items() if v}
    rep.rows_out = len(df)
    return df[schema.PRICES], rep


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    with open(path, encoding="utf-8-sig") as f:
        head = f.readline()
    if not any(d in head for d in ",;\t|"):  # single-column file: nothing to sniff
        return pd.read_csv(path)
    return pd.read_csv(path, sep=None, engine="python")


def ingest_prices(path: str | Path, mapping: Optional[ColumnMapping] = None) -> tuple[pd.DataFrame, ColumnMapping, CleaningReport]:
    """File -> canonical, cleaned prices. Filename stem is the ticker for single-instrument files."""
    raw = read_table(path)
    mapping = mapping or infer_mapping_heuristic(raw)
    canonical = apply_mapping(raw, mapping, default_ticker=Path(path).stem)
    cleaned, report = clean_prices(canonical)
    return cleaned, mapping, report
