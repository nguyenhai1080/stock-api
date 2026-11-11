"""Deterministic unit tests; no network or live market data required."""
import unittest
import numpy as np
import pandas as pd

from research.signals import scan_eod


def sample(rows=100):
    dates = pd.bdate_range("2026-01-01", periods=rows)
    j = np.arange(rows)
    # Mild upward trend with alternating closes to avoid persistent RSI overbought.
    close = 100 + 0.10 * j + 0.65 * np.where(j % 2 == 0, 1, -1)
    prices = pd.DataFrame({
        "date": dates, "ticker": "TEST", "open": close - 0.2,
        "high": close + 0.8, "low": close - 0.8,
        "close": close, "volume": np.full(rows, 2_000_000),
        "trading_value": np.full(rows, 20_000_000_000),
    })
    idx = pd.DataFrame({"date": dates, "close": 1000 + j * 0.4})
    return prices, idx


class ScannerTests(unittest.TestCase):
    def test_no_future_information(self):
        prices, idx = sample()
        whole = scan_eod(prices, idx)
        earlier = scan_eod(prices.iloc[:-1], idx.iloc[:-1])
        cols = ["breakout55", "pullback20", "rs_leader", "ma20", "rsi14"]
        pd.testing.assert_frame_equal(
            whole.iloc[:-1][cols].reset_index(drop=True),
            earlier[cols].reset_index(drop=True),
        )

    def test_gate_blocks_low_trading_value(self):
        prices, idx = sample()
        prices["trading_value"] = 1_000_000_000
        f = scan_eod(prices, idx)
        self.assertFalse(f["any_setup"].any())

    def test_gate_blocks_weak_index(self):
        prices, idx = sample()
        idx["close"] = 1200 - np.arange(len(idx)) * 2
        f = scan_eod(prices, idx)
        self.assertFalse(f["any_setup"].any())

    def test_insufficient_history_no_signals(self):
        prices, idx = sample(18)
        f = scan_eod(prices, idx)
        self.assertFalse(f["any_setup"].any())

    def test_rejects_duplicate_sessions(self):
        prices, idx = sample()
        with self.assertRaises(ValueError):
            scan_eod(pd.concat([prices, prices.iloc[[0]]]), idx)

    def test_rejects_bad_ohlc(self):
        prices, idx = sample()
        prices.loc[0, "low"] = prices.loc[0, "high"] + 2
        with self.assertRaises(ValueError):
            scan_eod(prices, idx)


if __name__ == "__main__":
    unittest.main()
