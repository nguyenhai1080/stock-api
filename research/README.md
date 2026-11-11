# V-Stock Research Signals v0.1

This research module is intentionally isolated from the existing FastAPI endpoints.

## Scope
- Point-in-time EOD calculation for three baseline **candidate** signals: 55-session breakout, pullback to MA20, and RS20 > VN-Index by 10 percentage points.
- VN-Index above MA50 gate; minimum average 20-session turnover of VND 10 billion; RSI < 75 and no more than 10% above MA20.
- Volume confirmation 1.3x prior-20-session average for breakout; RS20 uses simple 20-session returns.
- Returns all history rows with feature values and signal flags. Does not trade or assign purchase recommendations.

## Run tests
```bash
python -m unittest discover -s tests -p 'test_research_signals.py'
```

## Example
```python
from research.signals import scan_eod
result = scan_eod(prices_df, vnindex_df)
candidates = result.loc[result.any_setup]
```

`prices_df` columns: `date,ticker,open,high,low,close,volume,trading_value`. `vnindex_df` columns: `date,close`. Trading value must be **VND**, not billion VND. Dates need to be trading-session aligned.

## Important limitations
- This is **not** the implementation that generated the reported 430-trade backtest. Original trade records and the original signal logic have not been obtained.
- One-year input is insufficient for longer out-of-sample tests; obtain history across multiple regimes.
- The MA20 touch tolerance of 0.5 ATR and requirement that breakout be newly triggered are provisional choices, not verified definitions from the old engine.
- No point-in-time fundamentals, sector classifications, corporate-action adjustment, slippage, exchange-limit rules, position accounting, T+2 execution, or actual backtesting are implemented.
- Avoid using the signals as live trading instructions. Next step: reproduce the historical baseline using the original dataset and then implement execution and statistical evaluation.
