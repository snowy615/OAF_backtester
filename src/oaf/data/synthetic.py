"""Synthetic market with known properties, for demos and tests (no network, no licence).

The generated world deliberately contains the things a real dataset throws at a
backtester: late listings, delistings that follow a slide in price (so ignoring
them inflates results), index reconstitution, stock splits in the raw close, and
fundamentals published with a reporting lag. It also plants a modest 6-month
momentum effect and a 1-week reversal effect so example strategies have something
real to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .warehouse import Warehouse

SECTORS = ["tech", "financials", "healthcare", "energy", "consumer", "industrials", "utilities", "materials"]


def generate(n_tickers: int = 120, n_days: int = 2000, end: str = "2025-12-31", seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n_days)
    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
    sector_id = rng.integers(0, len(SECTORS), n_tickers)

    # life spans
    start_idx = np.where(rng.random(n_tickers) < 0.8, 0, rng.integers(0, n_days // 2, n_tickers))
    delists = rng.random(n_tickers) < 0.25
    life = rng.integers(300, n_days, n_tickers)
    end_idx = np.where(delists, np.minimum(start_idx + life, n_days - 1), n_days - 1)
    delists &= end_idx < n_days - 1
    distressed = delists & (rng.random(n_tickers) < 0.7)  # the rest are acquired at a premium

    beta = rng.normal(1.0, 0.25, n_tickers)
    idio_vol = rng.uniform(0.012, 0.028, n_tickers)
    mkt = rng.normal(0.0003, 0.009, n_days)
    sec = rng.normal(0.0, 0.005, (n_days, len(SECTORS)))

    logp = np.zeros((n_days, n_tickers))
    rets = np.zeros((n_days, n_tickers))
    for t in range(1, n_days):
        alpha = np.zeros(n_tickers)
        if t > 130:
            mom = logp[t - 6] - logp[t - 126]
            rev = logp[t - 1] - logp[t - 6]
            alpha = 0.0004 * _z(mom) - 0.0003 * _z(rev)
        slide = np.where(distressed & (end_idx - t < 120) & (t <= end_idx), -0.003, 0.0)
        r = beta * mkt[t] + sec[t, sector_id] + rng.normal(0, 1, n_tickers) * idio_vol + alpha + slide
        rets[t] = r
        logp[t] = logp[t - 1] + np.log1p(np.clip(r, -0.5, 0.5))

    adj = np.exp(logp) * rng.uniform(10, 200, n_tickers)
    listed = (np.arange(n_days)[:, None] >= start_idx) & (np.arange(n_days)[:, None] <= end_idx)

    # raw close = adjusted close / cumulative split factor (a few 2:1 splits)
    factor = np.ones((n_days, n_tickers))
    for j in rng.choice(n_tickers, 6, replace=False):
        k = rng.integers(n_days // 4, 3 * n_days // 4)
        factor[:k, j] = 0.5  # adj = raw * factor before the split
    raw = adj / factor

    shares = rng.lognormal(18, 1.0, n_tickers)
    noise = lambda s: rng.normal(0, s, (n_days, n_tickers))  # noqa: E731
    open_ = raw * np.exp(noise(0.004))
    high = np.maximum(raw, open_) * np.exp(np.abs(noise(0.006)))
    low = np.minimum(raw, open_) * np.exp(-np.abs(noise(0.006)))
    volume = np.round(shares * 0.004 * np.exp(noise(0.5)) / factor)

    frames = []
    for j, tk in enumerate(tickers):
        rows = listed[:, j]
        frames.append(
            pd.DataFrame(
                {
                    "date": dates[rows],
                    "ticker": tk,
                    "open": open_[rows, j],
                    "high": high[rows, j],
                    "low": low[rows, j],
                    "close": raw[rows, j],
                    "adj_close": adj[rows, j],
                    "volume": volume[rows, j],
                }
            )
        )
    prices = pd.concat(frames, ignore_index=True)

    meta = pd.DataFrame(
        {
            "ticker": tickers,
            "name": [f"Synthetic Co {i}" for i in range(n_tickers)],
            "sector": [SECTORS[s] for s in sector_id],
            "delisted_date": [dates[e] if d else pd.NaT for e, d in zip(end_idx, delists)],
            "delisting_return": [
                (rng.uniform(-0.9, -0.2) if dis else 0.15) if d else np.nan for d, dis in zip(delists, distressed)
            ],
        }
    )

    # Quarterly-reconstituted index of the largest listed names, as intervals.
    size = int(n_tickers * 0.6)
    cap = np.where(listed, adj * shares, np.nan)
    member = np.zeros((n_days, n_tickers), dtype=bool)
    current = np.zeros(n_tickers, dtype=bool)
    for t in range(n_days):
        if t % 63 == 0:
            order = np.argsort(-np.nan_to_num(cap[t], nan=-1))
            current = np.zeros(n_tickers, dtype=bool)
            current[order[:size]] = True
        member[t] = current & listed[t]
    membership = _intervals(member, dates, tickers, "demo_index")

    # Earnings yield, published 45 days after quarter end (the PIT trap).
    q_ends = pd.date_range(dates[0], dates[-1], freq="QE")
    ey = 0.05 + np.cumsum(rng.normal(0, 0.004, (len(q_ends), n_tickers)), axis=0) + rng.normal(0, 0.02, n_tickers)
    fundamentals = pd.DataFrame(
        {
            "ticker": np.repeat(tickers, len(q_ends)),
            "field": "earnings_yield",
            "period_end": np.tile(q_ends, n_tickers),
            "available_date": np.tile(q_ends + pd.Timedelta(days=45), n_tickers),
            "value": ey.T.ravel(),
        }
    )
    return {"prices": prices, "membership": membership, "fundamentals": fundamentals, "meta": meta}


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _intervals(member: np.ndarray, dates: pd.DatetimeIndex, tickers: list[str], universe: str) -> pd.DataFrame:
    rows = []
    for j, tk in enumerate(tickers):
        m = member[:, j].astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            still_in = b == len(m)
            rows.append((universe, tk, dates[a], pd.NaT if still_in else dates[b - 1]))
    return pd.DataFrame(rows, columns=["universe", "ticker", "start_date", "end_date"])


def write_demo_warehouse(root: str | Path, **kwargs) -> Warehouse:
    wh = Warehouse(root)
    data = generate(**kwargs)
    wh.write_prices(data["prices"], source="synthetic", replace=True)
    wh.write_membership(data["membership"], source="synthetic", replace=True)
    wh.write_fundamentals(data["fundamentals"], source="synthetic", replace=True)
    wh.write_meta(data["meta"], source="synthetic", replace=True)
    return wh
