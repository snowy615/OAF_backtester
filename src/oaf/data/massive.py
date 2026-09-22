"""Massive (formerly Polygon.io) market data -> warehouse tables.

Only the standard library is used for HTTP, so the client has no extra
dependencies. Every public ``fetch_*`` function returns canonical warehouse rows
(see :mod:`oaf.data.schema`); nothing here writes to disk.

What we take from Massive and how it maps:

- daily bars           ``/v2/aggs/ticker/{t}/range/1/day/{from}/{to}`` (per ticker) or
                       ``/v2/aggs/grouped/locale/us/market/stocks/{date}`` (whole market)
                       -> ``prices``. Raw (unadjusted) OHLCV is stored as-is and
                       ``adj_close`` is rebuilt from the splits table, so both series
                       are internally consistent.
- splits               ``/stocks/v1/splits`` -> split adjustment factors.
- tickers              ``/v3/reference/tickers`` with ``active=true`` *and* ``false``
                       -> ``meta`` (name, exchange, delisting date). Pulling the inactive
                       list is what keeps delisted names in the universe.
- financials           ``/stocks/financials/v1/income-statements`` and
                       ``/balance-sheets`` -> ``fundamentals``. ``period_end`` is what the
                       number describes; ``filing_date`` is when it became public, and
                       that is the date the warehouse keys on.

Set ``MASSIVE_API_KEY`` (``.env`` works). Plan limits: Basic is 5 calls/minute and 2
years of history; paid Stocks plans are unlimited calls. Financials need the Stocks
Advanced plan or the Financials & Ratios add-on.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import schema

BASE_URL = "https://api.massive.com"
FILING_LAG_CAP_DAYS = 90  # SEC 10-K outside deadline; see fetch_fundamentals
ENV_KEY = "MASSIVE_API_KEY"

# income-statement / balance-sheet keys -> warehouse fundamental field names
INCOME_FIELDS = {
    "revenue": "revenue",
    "consolidated_net_income_loss": "net_income",
    "basic_earnings_per_share": "eps",
    "diluted_earnings_per_share": "eps_diluted",
    "diluted_shares_outstanding": "shares_outstanding",
}
BALANCE_FIELDS = {
    "total_equity": "total_equity",
    "total_assets": "total_assets",
    "total_liabilities": "total_liabilities",
    "cash_and_equivalents": "cash",
}


class MassiveError(RuntimeError):
    pass


@dataclass
class MassiveClient:
    """Thin REST client: auth, retries, rate limiting and ``next_url`` pagination."""

    api_key: Optional[str] = None
    base_url: str = BASE_URL
    calls_per_minute: Optional[int] = None  # None = unlimited (paid plans); Basic plan is 5
    max_retries: int = 5
    timeout: float = 30.0
    # test seam: (url) -> parsed JSON. Replaces the network when set.
    transport: Optional[Callable[[str], dict]] = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get(ENV_KEY)
        if not self.api_key and self.transport is None:
            raise MassiveError(f"no Massive API key: set {ENV_KEY} in the environment or .env")
        self._last_calls: list[float] = []
        self._lock = threading.Lock()
        self.n_calls = 0

    # -- transport -----------------------------------------------------------
    def _throttle(self) -> None:
        if not self.calls_per_minute:
            return
        with self._lock:
            now = time.monotonic()
            self._last_calls = [t for t in self._last_calls if now - t < 60]
            if len(self._last_calls) >= self.calls_per_minute:
                time.sleep(60 - (now - self._last_calls[0]) + 0.05)
            self._last_calls.append(time.monotonic())

    def _fetch(self, url: str) -> dict:
        if self.transport is not None:
            self.n_calls += 1
            return self.transport(url)
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "oaf-backtester"})
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.n_calls += 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code in (401, 403):
                    raise MassiveError(f"Massive refused the request ({e.code}): {body}") from None
                if e.code == 429 or e.code >= 500:
                    if attempt == self.max_retries:
                        raise MassiveError(f"Massive error {e.code} after {attempt} retries: {body}") from None
                    retry_after = e.headers.get("Retry-After")
                    time.sleep(float(retry_after) if retry_after else delay)
                    delay = min(delay * 2, 30)
                    continue
                raise MassiveError(f"Massive error {e.code} for {url.split('?')[0]}: {body}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise MassiveError(f"could not reach Massive: {e}") from None
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise AssertionError("unreachable")

    def get(self, path: str, **params: Any) -> list[dict]:
        """GET ``path`` and follow ``next_url`` until exhausted. Returns all ``results``."""
        clean = {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in params.items() if v is not None}
        url = f"{self.base_url}{path}"
        if clean:
            url += "?" + urllib.parse.urlencode(clean)
        out: list[dict] = []
        while url:
            data = self._fetch(url)
            if data.get("status") not in (None, "OK", "DELAYED"):
                raise MassiveError(f"Massive returned {data.get('status')}: {data.get('error') or data.get('message')}")
            results = data.get("results") or []
            out.extend(results if isinstance(results, list) else [results])  # single-object endpoints
            url = data.get("next_url")
        return out


# -- helpers --------------------------------------------------------------------
def _day(ms: int) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date())


def _bars_frame(results: Iterable[dict], ticker: Optional[str] = None) -> pd.DataFrame:
    rows = [
        {
            "date": _day(r["t"]),
            "ticker": ticker or r["T"],
            "open": r.get("o"),
            "high": r.get("h"),
            "low": r.get("l"),
            "close": r.get("c"),
            "volume": r.get("v"),
        }
        for r in results
        if r.get("c") is not None
    ]
    return pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])


def split_factors(splits: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Cumulative factor so that ``close * factor`` is comparable with today's share count.

    A ``split_from -> split_to`` split executed on day D divides every price before D by
    ``split_to / split_from``. ``splits`` has columns ticker, execution_date, split_from,
    split_to.
    """
    factor = pd.Series(1.0, index=prices.index)
    if splits.empty:
        return factor
    for (ticker, exec_date, s_from, s_to) in splits[["ticker", "execution_date", "split_from", "split_to"]].itertuples(index=False):
        if not s_from or not s_to:
            continue
        before = (prices["ticker"] == ticker) & (prices["date"] < exec_date)
        factor[before] *= float(s_from) / float(s_to)
    return factor


def apply_splits(prices: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    out["adj_close"] = out["close"] * split_factors(splits, out)
    return out[schema.PRICES]


# -- public fetchers --------------------------------------------------------------
def fetch_splits(client: MassiveClient, tickers: Optional[Sequence[str]] = None, start: Optional[str] = None) -> pd.DataFrame:
    params: dict[str, Any] = {"limit": 5000, "sort": "execution_date.asc"}
    if start:
        params["execution_date.gte"] = start
    results: list[dict] = []
    if tickers and len(tickers) <= 100:
        results = client.get("/stocks/v1/splits", **params, **{"ticker.any_of": ",".join(tickers)})
    else:
        results = client.get("/stocks/v1/splits", **params)
    df = pd.DataFrame(results, columns=["ticker", "execution_date", "split_from", "split_to", "adjustment_type"])
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    if tickers:
        df = df[df["ticker"].isin(list(tickers))]
    return df.reset_index(drop=True)


def fetch_daily_bars(
    client: MassiveClient,
    tickers: Sequence[str],
    start: str,
    end: Optional[str] = None,
    splits: Optional[pd.DataFrame] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Unadjusted daily OHLCV per ticker, with ``adj_close`` rebuilt from splits."""
    end = end or date.today().isoformat()
    frames = []
    for i, t in enumerate(tickers):
        results = client.get(f"/v2/aggs/ticker/{t}/range/1/day/{start}/{end}", adjusted=False, sort="asc", limit=50000)
        frames.append(_bars_frame(results, t))
        if progress:
            progress(i + 1, len(tickers), t)
    prices = pd.concat(frames, ignore_index=True) if frames else _bars_frame([])
    if splits is None:
        splits = fetch_splits(client, tickers)
    return apply_splits(prices, splits)


def fetch_market(
    client: MassiveClient,
    start: str,
    end: Optional[str] = None,
    keep: Optional[Callable[[str], bool]] = None,
    splits: Optional[pd.DataFrame] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Every US stock, one grouped-daily call per session (unadjusted), then split-adjust."""
    days = pd.bdate_range(start, end or date.today())
    frames = []
    for i, d in enumerate(days):
        day = d.strftime("%Y-%m-%d")
        results = client.get(f"/v2/aggs/grouped/locale/us/market/stocks/{day}", adjusted=False)
        df = _bars_frame(results)
        if keep is not None and len(df):
            df = df[df["ticker"].map(keep)]
        frames.append(df)
        if progress:
            progress(i + 1, len(days), day)
    prices = pd.concat(frames, ignore_index=True) if frames else _bars_frame([])
    if splits is None:
        splits = fetch_splits(client, start=start)
    splits = splits[splits["ticker"].isin(prices["ticker"].unique())]
    return apply_splits(prices, splits)


def iter_market_years(
    client: MassiveClient,
    start: str,
    end: Optional[str] = None,
    keep: Optional[Callable[[str], bool]] = None,
    splits: Optional[pd.DataFrame] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    workers: int = 8,
):
    """Like :func:`fetch_market` but yields one split-adjusted frame per calendar year,
    so a decade of the whole market never has to sit in memory at once. Days within a
    year are fetched concurrently (paid plans have no call limit; set ``workers=1`` and
    ``calls_per_minute`` on the free plan)."""
    if splits is None:
        splits = fetch_splits(client, start=start)
    days = pd.bdate_range(start, end or date.today())
    done = 0

    def one_day(d):
        day = d.strftime("%Y-%m-%d")
        df = _bars_frame(client.get(f"/v2/aggs/grouped/locale/us/market/stocks/{day}", adjusted=False))
        return day, (df[df["ticker"].map(keep)] if keep is not None and len(df) else df)

    for year, chunk_days in days.groupby(days.year).items():
        frames = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for day, df in pool.map(one_day, chunk_days):
                frames.append(df)
                done += 1
                if progress:
                    progress(done, len(days), day)
        prices = pd.concat(frames, ignore_index=True) if frames else _bars_frame([])
        yield int(year), apply_splits(prices, splits[splits["ticker"].isin(prices["ticker"].unique())])


def readjust_history(prices: pd.DataFrame, splits: pd.DataFrame, since: str) -> pd.DataFrame:
    """After a refresh, fold splits executed on/after ``since`` into rows stored earlier.

    Stored rows already reflect every split before ``since``; only the new ones apply.
    """
    new_splits = splits[splits["execution_date"] >= pd.Timestamp(since)]
    if new_splits.empty:
        return prices
    out = prices.copy()
    old = out["date"] < pd.Timestamp(since)
    out.loc[old, "adj_close"] = out.loc[old, "adj_close"] * split_factors(new_splits, out)[old]
    return out


def fetch_tickers(client: MassiveClient, include_delisted: bool = True, types: Sequence[str] = ("CS",)) -> pd.DataFrame:
    """Reference list of US stocks -> ``meta`` rows (sector left blank; see ``fetch_sectors``)."""
    rows: list[dict] = []
    for active in ([True, False] if include_delisted else [True]):
        for typ in types:
            rows += client.get("/v3/reference/tickers", market="stocks", type=typ, active=active, limit=1000)
    if not rows:
        return pd.DataFrame(columns=schema.META + ["primary_exchange", "type"])
    df = pd.DataFrame(rows)
    delisted = pd.to_datetime(df.get("delisted_utc"), errors="coerce", utc=True)
    out = pd.DataFrame(
        {
            "ticker": df["ticker"].astype(str).str.upper(),
            "name": df.get("name", ""),
            "sector": np.nan,
            "delisted_date": delisted.dt.tz_localize(None).dt.normalize() if delisted is not None else pd.NaT,
            "delisting_return": np.nan,
            "primary_exchange": df.get("primary_exchange", ""),
            "type": df.get("type", ""),
        }
    )
    return out.drop_duplicates("ticker").reset_index(drop=True)


def fetch_sectors(client: MassiveClient, tickers: Sequence[str], progress: Optional[Callable] = None) -> pd.Series:
    """SIC description per ticker from the ticker-overview endpoint (one call each)."""
    out = {}
    for i, t in enumerate(tickers):
        try:
            res = client.get(f"/v3/reference/tickers/{t}")
        except MassiveError:
            res = []
        out[t] = res[0].get("sic_description") if res else None
        if progress:
            progress(i + 1, len(tickers), t)
    return pd.Series(out, name="sector")


def fetch_fundamentals(
    client: MassiveClient,
    tickers: Sequence[str],
    start: Optional[str] = None,
    progress: Optional[Callable] = None,
) -> pd.DataFrame:
    """Quarterly income-statement and balance-sheet items -> ``fundamentals`` rows.

    ``available_date`` = ``min(filing_date, period_end + FILING_LAG_CAP_DAYS)``. Massive's
    ``filing_date`` is the *latest* filing that contained a period - prior-year comparatives
    in a 10-K re-stamp a quarter about 13 months after it was first reported - which would
    make every number look a year stale. The cap is the SEC's outside deadline for the
    annual report, so a number is never made visible before it could have been filed.
    """
    rows = []
    batches = [list(tickers)[i : i + 50] for i in range(0, len(tickers), 50)]
    for i, batch in enumerate(batches):
        params: dict[str, Any] = {"tickers.any_of": ",".join(batch), "timeframe": "quarterly", "limit": 50000, "sort": "period_end.asc"}
        if start:
            params["period_end.gte"] = start
        for path, mapping in (("/stocks/financials/v1/income-statements", INCOME_FIELDS), ("/stocks/financials/v1/balance-sheets", BALANCE_FIELDS)):
            for rec in client.get(path, **params):
                tks = rec.get("tickers") or []
                for tk in tks:
                    if tk not in batch:
                        continue
                    for src, name in mapping.items():
                        val = rec.get(src)
                        if val is None or rec.get("filing_date") is None:
                            continue
                        rows.append((tk, name, rec["period_end"], rec["filing_date"], float(val)))
        if progress:
            progress(i + 1, len(batches), ",".join(batch[:3]) + ("..." if len(batch) > 3 else ""))
    df = pd.DataFrame(rows, columns=schema.FUNDAMENTALS)
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["available_date"] = pd.to_datetime(df["available_date"])
    cap = df["period_end"] + pd.Timedelta(days=FILING_LAG_CAP_DAYS)
    df["available_date"] = df["available_date"].where(df["available_date"] <= cap, cap)
    return df.drop_duplicates(["ticker", "field", "period_end", "available_date"]).reset_index(drop=True)


def membership_from_prices(prices: pd.DataFrame, meta: pd.DataFrame, universe: str) -> pd.DataFrame:
    """One interval per ticker: first bar -> delisting date (or open-ended)."""
    first = prices.groupby("ticker")["date"].min()
    delisted = meta.set_index("ticker")["delisted_date"] if len(meta) else pd.Series(dtype="datetime64[ns]")
    return pd.DataFrame(
        {
            "universe": universe,
            "ticker": first.index,
            "start_date": first.values,
            "end_date": delisted.reindex(first.index).values,
        }
    ).reset_index(drop=True)
