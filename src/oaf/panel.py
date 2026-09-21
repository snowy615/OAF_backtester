"""In-memory market data handed to the signal engine and the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Panel:
    """Aligned (dates x tickers) frames.

    ``fields``   name -> DataFrame. Always contains ``close`` and ``returns``.
    ``mask``     True where a ticker is a tradable universe member on that date
                 (point-in-time: membership as known on the day, price present).
    ``groups``   name -> Series(ticker -> label), e.g. ``sector``.
    """

    fields: dict[str, pd.DataFrame]
    mask: pd.DataFrame
    groups: dict[str, pd.Series] = field(default_factory=dict)
    universe: str = "all"

    def __post_init__(self) -> None:
        for req in ("close", "returns"):
            if req not in self.fields:
                raise ValueError(f"Panel is missing required field {req!r}")
        ref = self.fields["close"]
        for name, df in self.fields.items():
            if not (df.index.equals(ref.index) and df.columns.equals(ref.columns)):
                raise ValueError(f"field {name!r} is not aligned with 'close'")
        self.mask = self.mask.reindex(index=ref.index, columns=ref.columns).fillna(False).astype(bool)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields["close"].index

    @property
    def tickers(self) -> pd.Index:
        return self.fields["close"].columns

    @property
    def returns(self) -> pd.DataFrame:
        return self.fields["returns"]

    def field_names(self) -> list[str]:
        return sorted(self.fields)

    def slice(self, start=None, end=None) -> "Panel":
        return Panel(
            fields={k: v.loc[start:end] for k, v in self.fields.items()},
            mask=self.mask.loc[start:end],
            groups=self.groups,
            universe=self.universe,
        )

    def restrict(self, tickers: list[str]) -> "Panel":
        keep = [t for t in self.tickers if t in set(tickers)]
        if not keep:
            raise ValueError("none of the requested tickers are in the panel")
        return Panel(
            fields={k: v[keep] for k, v in self.fields.items()},
            mask=self.mask[keep],
            groups={k: g.reindex(keep) for k, g in self.groups.items()},
            universe=self.universe,
        )
