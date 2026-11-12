"""Conservative daily-bar trade simulation for V-Stock research.

No live orders. Prices must be on a consistent basis; for realistic P&L use
tradeable/unadjusted execution quotes and account separately for corporate
actions. EOD signals are entered no earlier than the next trading session.
"""
from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    stop_atr: float = 2.0
    target_atr: float = 3.0
    max_hold_sessions: int = 10
    earliest_sell_session: int = 2  # entry=0; earliest possible sell=entry+2, conservative
    max_entry_gap_pct: float = 0.05
    buy_fee: float = 0.0015
    sell_fee: float = 0.0015
    sell_tax: float = 0.001
    slippage: float = 0.001


def _valid_bar(row):
    o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
    return all(math.isfinite(v) and v > 0 for v in (o, h, l, c)) and l <= min(o, c) <= max(o, c) <= h


def simulate_one(prices: pd.DataFrame, signal_date, signal_close: float,
                 signal_atr: float, *, config: ExecutionConfig = ExecutionConfig()):
    """Return closed trade or an explicit unfilled/unfinished status.

    A stop crossed before sell eligibility is not considered executed. Ambiguous
    stop-and-target bars are filled at stop. Stops and targets are assumed
    executable when eligible; price limits, auctions and order-book depth still
    require a richer execution model before using performance operationally.
    """
    needed = {"date", "open", "high", "low", "close"}
    if not needed.issubset(prices.columns):
        raise ValueError(f"Missing columns: {sorted(needed - set(prices.columns))}")
    if config.max_hold_sessions < 1 or config.earliest_sell_session < 1:
        raise ValueError("Invalid holding period or earliest sell session")
    if not (math.isfinite(signal_close) and signal_close > 0 and
            math.isfinite(signal_atr) and signal_atr > 0):
        raise ValueError("Invalid signal price or ATR")

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").reset_index(drop=True)
    if p["date"].duplicated().any():
        raise ValueError("Duplicate trading dates")
    if not p.apply(_valid_bar, axis=1).all():
        raise ValueError("Invalid OHLC bar")

    idxs = p.index[p["date"] > pd.Timestamp(signal_date)]
    if len(idxs) == 0:
        return {"status": "NO_ENTRY_BAR"}
    e = int(idxs[0])
    o = float(p.iloc[e]["open"])
    if o > signal_close * (1 + config.max_entry_gap_pct):
        return {"status": "ENTRY_GAP_REJECTED"}
    entry = o * (1 + config.slippage)
    stop = entry - config.stop_atr * signal_atr
    target = entry + config.target_atr * signal_atr
    if stop <= 0:
        return {"status": "INVALID_STOP"}

    last = min(len(p) - 1, e + config.max_hold_sessions - 1)
    first_sell = e + config.earliest_sell_session
    breached_while_locked = False
    if last < first_sell:
        return {"status": "INSUFFICIENT_EXIT_HISTORY"}

    # Track MFE and MAE from entry onward, including the restricted sell window.
    mfe, mae = 0.0, 0.0
    for i in range(e, last + 1):
        bar = p.iloc[i]
        hi, lo = float(bar["high"]), float(bar["low"])
        mfe = max(mfe, hi / entry - 1)
        mae = min(mae, lo / entry - 1)
        if i < first_sell:
            if lo <= stop:
                breached_while_locked = True
            continue

        op, cl = float(bar["open"]), float(bar["close"])
        ambiguous = lo <= stop and hi >= target and stop < op < target
        if op <= stop:
            raw_exit, reason = op, "STOP_GAP"
        elif op >= target:
            raw_exit, reason = target, "TARGET_GAP"
        elif lo <= stop:
            raw_exit, reason = stop, "STOP"
        elif hi >= target:
            raw_exit, reason = target, "TARGET"
        elif i == last:
            raw_exit, reason = cl, "TIME"
        else:
            continue

        sell_price = raw_exit * (1 - config.slippage)
        invested = entry * (1 + config.buy_fee)
        proceeds = sell_price * (1 - config.sell_fee - config.sell_tax)
        return {
            "status": "CLOSED",
            "signal_date": str(pd.Timestamp(signal_date).date()),
            "entry_date": str(p.iloc[e]["date"].date()),
            "exit_date": str(bar["date"].date()),
            "entry_price": entry,
            "exit_price": sell_price,
            "stop": stop,
            "target": target,
            "net_return": proceeds / invested - 1,
            "holding_sessions": i - e + 1,
            "exit_reason": reason,
            "ambiguous_bar": bool(ambiguous),
            "stop_breached_while_locked": breached_while_locked,
            "mfe": mfe,
            "mae": mae,
        }
    return {"status": "NO_EXIT"}


def backtest_signals(features: pd.DataFrame, *,
                     config: ExecutionConfig = ExecutionConfig()) -> pd.DataFrame:
    """Backtest unique (ticker, date) setups with no overlapping positions.

    Expects features from scan_eod(); each ticker needs a complete OHLC price
    history extending beyond the final candidate date. One position/ticker at
    a time, unlimited capital across tickers; NOT a portfolio-level backtest.
    """
    required = {"date", "ticker", "open", "high", "low", "close", "atr14",
                "any_setup", "breakout55", "pullback20", "rs_leader"}
    if not required.issubset(features.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(features.columns))}")
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    results = []
    for ticker, g in frame.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        unavailable_until = pd.Timestamp.min
        for _, s in g[g["any_setup"]].iterrows():
            if s["date"] <= unavailable_until or pd.isna(s["atr14"]):
                continue
            trade = simulate_one(g, s["date"], float(s["close"]), float(s["atr14"]), config=config)
            if trade["status"] == "CLOSED":
                unavailable_until = pd.Timestamp(trade["exit_date"])
            elif trade["status"] == "NO_EXIT":
                unavailable_until = g.iloc[-1]["date"]
            results.append({
                "ticker": ticker,
                "setups": "|".join(k for k in ("breakout55", "pullback20", "rs_leader") if s[k]),
                **trade,
            })
    return pd.DataFrame(results)
