"""Optional market-data downloaders. Each returns canonical ``prices`` rows."""

from __future__ import annotations

import warnings
from typing import Sequence

import pandas as pd

from . import schema


def fetch_yahoo(tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance (``pip install 'oaf-backtester[yahoo]'``).

    Yahoo only serves tickers that still exist, so anything built from a current
    constituent list is survivorship-biased. Fine for prototyping a pitch; pair it
    with a point-in-time membership table before trusting the numbers.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance is not installed: pip install 'oaf-backtester[yahoo]'") from e

    warnings.warn(
        "Yahoo data contains survivors only; results on it are biased upwards unless you also "
        "load delisted names and point-in-time membership.",
        stacklevel=2,
    )
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=False, group_by="ticker", progress=False)
    frames = []
    for tk in tickers:
        sub = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
        sub = sub.rename(columns=lambda c: str(c).lower().replace(" ", "_")).reset_index()
        sub = sub.rename(columns={sub.columns[0]: "date"})
        sub["ticker"] = tk.upper()
        frames.append(sub.dropna(subset=["close"]))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out[schema.PRICES]
