"""Load index / universe constituent lists into the point-in-time membership table.

Massive does not publish index constituents, so membership comes from a file. Two
layouts are accepted:

- intervals: columns ``ticker, start_date[, end_date]`` (blank end = still a member).
- snapshots: columns ``date, ticker`` - one row per (as-of date, member). Consecutive
  snapshots are turned into intervals: a ticker is a member from the first snapshot
  that lists it until the day before the first later snapshot that does not.

A single-column file of tickers is treated as a snapshot dated today, which is
survivorship-biased; the loader says so.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from .ingest import read_table


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def load_membership(path: str | Path, universe: str) -> pd.DataFrame:
    df = read_table(path)
    df.columns = [_norm(c) for c in df.columns]
    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    if "ticker" not in df.columns:
        if df.shape[1] == 1:
            df = df.rename(columns={df.columns[0]: "ticker"})
        else:
            raise ValueError(f"{path}: need a 'ticker' column; found {list(df.columns)}")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    if "start_date" in df.columns:
        out = pd.DataFrame(
            {
                "universe": universe,
                "ticker": df["ticker"],
                "start_date": pd.to_datetime(df["start_date"]),
                "end_date": pd.to_datetime(df["end_date"]) if "end_date" in df.columns else pd.NaT,
            }
        )
        return out.dropna(subset=["start_date"]).reset_index(drop=True)

    if "date" not in df.columns:
        warnings.warn(
            f"{path} has tickers only: treated as today's members, which is survivorship-biased. "
            "Provide start_date/end_date intervals or dated snapshots for a point-in-time universe.",
            stacklevel=2,
        )
        df["date"] = pd.Timestamp.today().normalize()
    df["date"] = pd.to_datetime(df["date"])
    return _snapshots_to_intervals(df[["date", "ticker"]], universe)


def _snapshots_to_intervals(snap: pd.DataFrame, universe: str) -> pd.DataFrame:
    dates = sorted(snap["date"].unique())
    members = {d: set(snap.loc[snap["date"] == d, "ticker"]) for d in dates}
    open_since: dict[str, pd.Timestamp] = {}
    rows = []
    for i, d in enumerate(dates):
        for t in members[d]:
            open_since.setdefault(t, d)
        gone = [t for t in open_since if t not in members[d]]
        for t in gone:
            rows.append((universe, t, open_since.pop(t), d - pd.Timedelta(days=1)))
    for t, since in open_since.items():
        rows.append((universe, t, since, pd.NaT))
    out = pd.DataFrame(rows, columns=["universe", "ticker", "start_date", "end_date"])
    return out.sort_values(["ticker", "start_date"]).reset_index(drop=True)
