"""Claude-assisted data ingestion: read a messy file's layout, emit a ``ColumnMapping``.

Claude only ever sees the header and a small sample of rows, and only ever decides
*how columns map onto the warehouse schema*. The transformation and cleaning are
done by deterministic code in ``oaf.data.ingest``, and the mapping is saved next to
the data so a human can review or correct it.
"""

from __future__ import annotations

import pandas as pd

from ..data.ingest import ColumnMapping
from .client import get_client, parse_structured

SAMPLE_ROWS = 15

SYSTEM = """\
You map raw market-data files onto the Oxford Alpha Fund warehouse's canonical daily \
price schema: date, ticker, open, high, low, close, adj_close, volume. You are given a \
file name, its column names with dtypes, and its first rows. Return a ColumnMapping.

- `layout` is "long" when each row is one (date, ticker) observation, and "wide" when \
each row is a date and each remaining column is a ticker holding one price field. For wide \
files set `wide_field` to "adj_close" if the prices look adjusted (or you cannot tell) and \
"close" if they are clearly raw.
- Column fields (`date_col`, `ticker_col`, `open_col`, ...) must be exact column names \
from the file, or null when the file has no such column. When a file has a single price \
column, map it to `close_col`, and also to `adj_close_col` only if it is explicitly adjusted.
- If there is no ticker column because the file covers one instrument, set `ticker_value` \
from the file name or contents.
- Set `date_format` (strptime) only when the format is unambiguous from the sample; set \
`dayfirst` true for day-first dates such as 31/01/2024. Otherwise leave both at defaults.
- `price_scale` converts quoted prices to major currency units: 0.01 for London prices \
quoted in pence (GBX/GBp), otherwise 1.0.
- Use `notes` for anything a human should check before trusting the data: suspected \
unadjusted prices, mixed currencies, intraday timestamps, footer rows, and so on.
"""


def infer_mapping_llm(df: pd.DataFrame, filename: str = "", client=None) -> ColumnMapping:
    client = client or get_client()
    dtypes = ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items())
    sample = df.head(SAMPLE_ROWS).to_csv(index=False)
    content = f"File: {filename}\nRows: {len(df)}\nColumns: {dtypes}\n\nFirst {SAMPLE_ROWS} rows:\n{sample}"
    mapping, _ = parse_structured(client, SYSTEM, [{"role": "user", "content": content}], ColumnMapping, max_tokens=8000)

    known = set(df.columns)
    for name in ("date_col", "ticker_col", "open_col", "high_col", "low_col", "close_col", "adj_close_col", "volume_col"):
        col = getattr(mapping, name)
        if col is not None and col not in known:
            raise ValueError(f"Claude mapped {name} to {col!r}, which is not a column of {filename or 'the file'}")
    return mapping
