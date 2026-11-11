"""Research-only EOD signal scanner for V-Stock.

No orders, recommendations or production API changes. Each signal row is
computed exclusively from information available at that day's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = {"date", "ticker", "open", "high", "low", "close", "volume", "trading_value"}


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    return rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)


def _features_one_stock(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, h, l, v = (g[k].astype(float) for k in ("close", "high", "low", "volume"))
    g["ma20"] = c.rolling(20, min_periods=20).mean()
    g["ma50"] = c.rolling(50, min_periods=50).mean()
    g["ma50_10ago"] = g["ma50"].shift(10)
    g["high55_prev"] = h.shift(1).rolling(55, min_periods=55).max()
    g["high55_prev_yesterday"] = g["high55_prev"].shift(1)
    g["vol20_prev"] = v.shift(1).rolling(20, min_periods=20).mean()
    g["rvol"] = v / g["vol20_prev"].replace(0, np.nan)
    g["value20"] = g["trading_value"].astype(float).rolling(20, min_periods=20).mean()
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    g["rsi14"] = _wilder_rsi(c)
    g["ret20"] = c.pct_change(20, fill_method=None)
    return g


def scan_eod(prices: pd.DataFrame, vnindex: pd.DataFrame, *,
             min_trading_value: float = 10_000_000_000,
             pullback_atr_tolerance: float = 0.5) -> pd.DataFrame:
    """Return historical flags with no future data.

    Index must contain 'date' and 'close'. Assumptions: both datasets use
    session-aligned dates; trading_value is expressed in VND; OHLC values are
    consistent (same adjustment basis). Rows with unavailable features do
    not trigger signals. A green candle means close > open, as initially specified.
    """
    missing = REQUIRED.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    if not {"date", "close"}.issubset(vnindex.columns):
        raise ValueError("VN-Index requires date and close")
    p = prices.copy()
    idx = vnindex.copy()
    p["date"] = pd.to_datetime(p["date"])
    idx["date"] = pd.to_datetime(idx["date"])
    if p.duplicated(["ticker", "date"]).any() or idx.duplicated(["date"]).any():
        raise ValueError("Duplicate ticker-date or index date")
    for col in ("open", "high", "low", "close", "volume", "trading_value"):
        p[col] = pd.to_numeric(p[col], errors="raise")
    if (p[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Nonpositive OHLC")
    if ((p["high"] < p[["open", "close", "low"]].max(axis=1)) |
        (p["low"] > p[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC")
    if (p[["volume", "trading_value"]] < 0).any().any():
        raise ValueError("Negative volume/trading value")
    idx = idx.sort_values("date")
    idx["idx_ma50"] = idx["close"].rolling(50, min_periods=50).mean()
    idx["idx_ret20"] = idx["close"].pct_change(20, fill_method=None)
    idx = idx.rename(columns={"close": "index_close"})
    features = pd.concat(
        [_features_one_stock(g) for _, g in p.groupby("ticker", sort=False)],
        ignore_index=True,
    )
    features = features.merge(
        idx[["date", "index_close", "idx_ma50", "idx_ret20"]],
        on="date", how="left", validate="many_to_one",
    )
    f = features
    f["rs20_pp"] = (f["ret20"] - f["idx_ret20"]) * 100
    market = f["index_close"] > f["idx_ma50"]
    liquid = f["value20"] >= min_trading_value
    safe = (f["rsi14"] < 75) & (f["close"] <= 1.10 * f["ma20"])
    trend = ((f["close"] > f["ma20"]) &
             (f["ma20"] > f["ma50"]) &
             (f["ma50"] > f["ma50_10ago"]))
    breakout = ((f["close"] > f["high55_prev"]) &
                (f["close"].groupby(f["ticker"]).shift(1) <= f["high55_prev_yesterday"]) &
                (f["rvol"] >= 1.3))
    pullback = (trend &
                (f["low"] <= f["ma20"] + pullback_atr_tolerance * f["atr14"]) &
                (f["close"] > f["open"]))
    strength = (trend & (f["rs20_pp"] > 10))
    gate = market & liquid & safe
    f["breakout55"] = (gate & breakout).fillna(False)
    f["pullback20"] = (gate & pullback).fillna(False)
    f["rs_leader"] = (gate & strength).fillna(False)
    f["any_setup"] = f[["breakout55", "pullback20", "rs_leader"]].any(axis=1)
    return f.sort_values(["date", "ticker"]).reset_index(drop=True)
