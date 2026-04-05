from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any
import math
import pandas as pd

app = FastAPI(title="vn-signal-bridge", version="2.0.0")

SUPPORTED_SOURCES = ["KBS", "MSN", "FMP", "VCI"]
MAX_WORKERS = 8
CACHE_TTL_SECONDS = 900  # 15 minutes


class BatchRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)


class CacheEntry(BaseModel):
    expires_at: float
    data: dict[str, Any]


# ----------------------------
# Static mappings
# ----------------------------
SECTOR_MAP = {
    "VCB": ("Bank", "VN30"),
    "BID": ("Bank", "VN30"),
    "CTG": ("Bank", "VN30"),
    "TCB": ("Bank", "VN30"),
    "MBB": ("Bank", "VN30"),
    "ACB": ("Bank", "LargeCap"),
    "VPB": ("Bank", "VN30"),
    "HDB": ("Bank", "LargeCap"),
    "STB": ("Bank", "LargeCap"),

    "SSI": ("Broker", "VN30"),
    "VND": ("Broker", "LargeCap"),
    "VIX": ("Broker", "Midcap"),
    "HCM": ("Broker", "LargeCap"),
    "SHS": ("Broker", "Midcap"),
    "MBS": ("Broker", "Midcap"),
    "BSI": ("Broker", "Midcap"),
    "FTS": ("Broker", "Midcap"),
    "AGR": ("Broker", "Midcap"),

    "VIC": ("RealEstate", "VN30"),
    "VHM": ("RealEstate", "VN30"),
    "NVL": ("RealEstate", "LargeCap"),
    "PDR": ("RealEstate", "Midcap"),
    "KDH": ("RealEstate", "LargeCap"),
    "NLG": ("RealEstate", "LargeCap"),
    "DXG": ("RealEstate", "Midcap"),
    "DIG": ("RealEstate", "Midcap"),
    "CEO": ("RealEstate", "Midcap"),
    "IDC": ("IndustrialZone", "Midcap"),

    "HPG": ("Steel", "VN30"),
    "HSG": ("Steel", "Midcap"),
    "NKG": ("Steel", "Midcap"),

    "MWG": ("Retail", "VN30"),
    "PNJ": ("Retail", "LargeCap"),
    "FRT": ("Retail", "Midcap"),
    "DGW": ("Retail", "Midcap"),
    "CRC": ("Retail", "Midcap"),

    "GAS": ("Energy", "VN30"),
    "POW": ("Energy", "VN30"),
    "PVD": ("OilGas", "Midcap"),
    "PVS": ("OilGas", "Midcap"),
    "BSR": ("OilGas", "Midcap"),
    "OIL": ("OilGas", "Midcap"),

    "GMD": ("Logistics", "LargeCap"),
    "HAH": ("Logistics", "Midcap"),
    "VSC": ("Logistics", "Midcap"),

    "DGC": ("Chemicals", "LargeCap"),
    "DPM": ("Chemicals", "Midcap"),
    "DCM": ("Chemicals", "Midcap"),
    "CSV": ("Chemicals", "Midcap"),

    "TCM": ("Textile", "Midcap"),
    "MSH": ("Textile", "Midcap"),
    "STK": ("Textile", "Midcap"),

    "VHC": ("Seafood", "Midcap"),
    "ANV": ("Seafood", "Midcap"),
    "FMC": ("Seafood", "Midcap"),

    "CTD": ("Construction", "Midcap"),
    "HBC": ("Construction", "Midcap"),
    "FCN": ("Construction", "Midcap"),
    "HHV": ("Infrastructure", "Midcap"),

    "FPT": ("Technology", "VN30"),
    "CMG": ("Technology", "Midcap"),
}


# ----------------------------
# In-memory TTL cache
# ----------------------------
_quote_cache: dict[str, CacheEntry] = {}
_cache_lock = Lock()


def now_ts() -> float:
    return datetime.utcnow().timestamp()


def cache_get(key: str):
    with _cache_lock:
        entry = _quote_cache.get(key)
        if not entry:
            return None
        if entry.expires_at < now_ts():
            _quote_cache.pop(key, None)
            return None
        return entry.data


def cache_set(key: str, data: dict[str, Any], ttl_seconds: int = CACHE_TTL_SECONDS):
    with _cache_lock:
        _quote_cache[key] = CacheEntry(
            expires_at=now_ts() + ttl_seconds,
            data=data
        )


# ----------------------------
# Utility
# ----------------------------
def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def unique_symbols(symbols: list[str]) -> list[str]:
    seen = set()
    out = []
    for s in symbols:
        code = normalize_symbol(s)
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def infer_sector_and_group(symbol: str):
    code = normalize_symbol(symbol)
    return SECTOR_MAP.get(code, ("Other", "General"))


def classify_liquidity(vol_ma20: float):
    if vol_ma20 >= 1_000_000:
        return "High"
    if vol_ma20 >= 300_000:
        return "Medium"
    return "Low"


# ----------------------------
# Indicators
# ----------------------------
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def get_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_indicators(df: pd.DataFrame) -> dict:
    df = df.copy()

    close_col = get_first_existing_column(df, ["close", "Close", "c"])
    volume_col = get_first_existing_column(df, ["volume", "Volume", "v"])
    high_col = get_first_existing_column(df, ["high", "High", "h"])

    if not close_col:
        raise ValueError("No close column found")

    df["close_x"] = pd.to_numeric(df[close_col], errors="coerce")
    df["volume_x"] = pd.to_numeric(df[volume_col], errors="coerce") if volume_col else 0
    df["high_x"] = pd.to_numeric(df[high_col], errors="coerce") if high_col else df["close_x"]

    df = df.dropna(subset=["close_x"]).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("Insufficient clean rows after normalization")

    df["ma20"] = df["close_x"].rolling(20, min_periods=20).mean()
    df["ma50"] = df["close_x"].rolling(50, min_periods=50).mean()
    df["ma200"] = df["close_x"].rolling(200, min_periods=200).mean()
    df["vol_ma20"] = df["volume_x"].rolling(20, min_periods=20).mean()
    df["rsi14"] = calc_rsi(df["close_x"], 14)
    df["high_20_prev"] = df["high_x"].rolling(20, min_periods=20).max().shift(1)

    last = df.iloc[-1]

    current_price = safe_float(last["close_x"])
    ma20 = safe_float(last["ma20"])
    ma50 = safe_float(last["ma50"])
    ma200 = safe_float(last["ma200"])
    rsi14 = safe_float(last["rsi14"])
    volume = safe_float(last["volume_x"])
    vol_ma20 = safe_float(last["vol_ma20"])
    breakout_level = safe_float(last["high_20_prev"])

    pct_from_ma20 = ((current_price / ma20) - 1) * 100 if ma20 > 0 else 0
    pct_from_ma50 = ((current_price / ma50) - 1) * 100 if ma50 > 0 else 0
    pct_from_ma200 = ((current_price / ma200) - 1) * 100 if ma200 > 0 else 0
    breakout_distance_pct = ((current_price / breakout_level) - 1) * 100 if breakout_level > 0 else 0
    volume_ratio = (volume / vol_ma20) if vol_ma20 > 0 else 0

    trend_stage = "Unknown"
    if ma20 > 0 and ma50 > 0 and ma200 > 0:
        if current_price > ma20 > ma50 > ma200:
            trend_stage = "StrongUptrend"
        elif current_price > ma20 and ma20 > ma50:
            trend_stage = "Uptrend"
        elif current_price < ma20 < ma50 < ma200:
            trend_stage = "StrongDowntrend"
        elif current_price < ma20 and ma20 < ma50:
            trend_stage = "Downtrend"
        else:
            trend_stage = "Sideways"

    return {
        "currentPrice": current_price,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi14": rsi14,
        "volume": volume,
        "volMa20": vol_ma20,
        "volumeRatio": volume_ratio,
        "breakoutLevel": breakout_level,
        "pctFromMa20": pct_from_ma20,
        "pctFromMa50": pct_from_ma50,
        "pctFromMa200": pct_from_ma200,
        "breakoutDistancePct": breakout_distance_pct,
        "trendStage": trend_stage,
    }


def compute_signal_score(payload: dict) -> float:
    price = safe_float(payload.get("currentPrice"))
    ma20 = safe_float(payload.get("ma20"))
    ma50 = safe_float(payload.get("ma50"))
    ma200 = safe_float(payload.get("ma200"))
    rsi = safe_float(payload.get("rsi14"))
    vol_ratio = safe_float(payload.get("volumeRatio"))
    breakout_dist = safe_float(payload.get("breakoutDistancePct"))

    score = 0.0

    if price > 0 and ma20 > 0 and price > ma20:
        score += 15
    if ma20 > 0 and ma50 > 0 and ma20 > ma50:
        score += 20
    if ma50 > 0 and ma200 > 0 and ma50 > ma200:
        score += 25

    if 45 <= rsi <= 70:
        score += 15
    elif 70 < rsi <= 78:
        score += 5
    elif rsi < 35:
        score -= 10

    if vol_ratio >= 1.5:
        score += 15
    elif vol_ratio >= 1.1:
        score += 8

    if -3 <= breakout_dist <= 2:
        score += 10
    elif breakout_dist < -8:
        score -= 5

    return round(max(score, 0), 2)


def classify_setup(score: float) -> str:
    if score >= 75:
        return "A"
    if score >= 55:
        return "B"
    if score >= 35:
        return "C"
    return "D"


# ----------------------------
# Data fetch
# ----------------------------
def fetch_stock_df(symbol: str, source: str):
    from vnstock import Vnstock

    stock = Vnstock().stock(symbol=symbol, source=source)
    start_date = (datetime.now() - timedelta(days=420)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    df = stock.quote.history(
        start=start_date,
        end=end_date,
        interval="1D"
    )
    return df


def get_signal_payload(symbol: str):
    code = normalize_symbol(symbol)
    if not code:
        return {
            "ok": False,
            "symbol": "",
            "error": "Invalid symbol",
            "details": []
        }

    cached = cache_get(f"signal:{code}")
    if cached:
        return cached

    errors = []

    for source in SUPPORTED_SOURCES:
        try:
            df = fetch_stock_df(code, source)

            if df is None or len(df) < 30:
                errors.append(f"{source}: insufficient data")
                continue

            indicators = build_indicators(df)
            result = {
                "ok": True,
                "symbol": code,
                "source": source,
                "priceDate": datetime.now().strftime("%Y-%m-%d"),
                **indicators
            }
            result["signalScore"] = compute_signal_score(result)
            result["setupRank"] = classify_setup(result["signalScore"])

            cache_set(f"signal:{code}", result)
            return result

        except Exception as e:
            errors.append(f"{source}: {str(e)}")

    result = {
        "ok": False,
        "symbol": code,
        "error": "All sources failed",
        "details": errors
    }
    cache_set(f"signal:{code}", result, ttl_seconds=120)
    return result


def build_universe_item(code: str) -> dict:
    payload = get_signal_payload(code)
    sector, group_tag = infer_sector_and_group(code)

    if payload.get("ok"):
        vol_ma20 = safe_float(payload.get("volMa20"))
        liquidity_tier = classify_liquidity(vol_ma20)
        active = liquidity_tier in ["High", "Medium"]

        return {
            "ok": True,
            "symbol": code,
            "sector": sector,
            "groupTag": group_tag,
            "liquidityTier": liquidity_tier,
            "active": active,
            "trendStage": payload.get("trendStage", "Unknown"),
            "setupRank": payload.get("setupRank", "D"),
            "signalScore": payload.get("signalScore", 0),
            "currentPrice": payload.get("currentPrice", 0),
            "volume": payload.get("volume", 0),
            "volMa20": payload.get("volMa20", 0),
            "volumeRatio": payload.get("volumeRatio", 0),
            "rsi14": payload.get("rsi14", 0),
            "ma20": payload.get("ma20", 0),
            "ma50": payload.get("ma50", 0),
            "ma200": payload.get("ma200", 0),
            "breakoutLevel": payload.get("breakoutLevel", 0),
            "breakoutDistancePct": payload.get("breakoutDistancePct", 0),
            "pctFromMa20": payload.get("pctFromMa20", 0),
            "pctFromMa50": payload.get("pctFromMa50", 0),
            "pctFromMa200": payload.get("pctFromMa200", 0),
            "source": payload.get("source", ""),
            "priceDate": payload.get("priceDate", ""),
        }

    return {
        "ok": False,
        "symbol": code,
        "sector": sector,
        "groupTag": group_tag,
        "liquidityTier": "Low",
        "active": False,
        "trendStage": "Unknown",
        "setupRank": "D",
        "signalScore": 0,
        "error": payload.get("error", "No data"),
        "details": payload.get("details", [])
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "vn-signal-bridge", "version": "2.0.0"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "healthy",
        "cacheItems": len(_quote_cache),
        "supportedSources": SUPPORTED_SOURCES
    }


@app.get("/signal")
def signal(symbol: str):
    return get_signal_payload(symbol)


@app.post("/batch_signals")
def batch_signals(req: BatchRequest):
    symbols = unique_symbols(req.symbols)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_signal_payload, code): code for code in symbols}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x.get("symbol", ""))
    success = sum(1 for x in results if x.get("ok"))

    return {
        "ok": True,
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "results": results
    }


@app.post("/batch_universe")
def batch_universe(req: BatchRequest):
    try:
        symbols = unique_symbols(req.symbols)
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(build_universe_item, code): code for code in symbols}
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(
            key=lambda x: (
                0 if x.get("ok") else 1,
                -safe_float(x.get("signalScore", 0)),
                x.get("symbol", "")
            )
        )

        success = sum(1 for x in results if x.get("ok"))

        return {
            "ok": True,
            "total": len(results),
            "success": success,
            "failed": len(results) - success,
            "results": results
        }

    except Exception as e:
        import traceback
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/universe_top")
def universe_top(limit: int = 50, active_only: bool = True):
    """
    Optional endpoint:
    - expects caller to have warmed the cache or to adapt later with a predefined universe list
    """
    items = []

    with _cache_lock:
        for _, entry in _quote_cache.items():
            data = entry.data
            if not isinstance(data, dict):
                continue
            if data.get("ok") and data.get("symbol"):
                code = data["symbol"]
                items.append(build_universe_item(code))

    if active_only:
        items = [x for x in items if x.get("active")]

    items.sort(
        key=lambda x: (
            0 if x.get("ok") else 1,
            -safe_float(x.get("signalScore", 0)),
            x.get("symbol", "")
        )
    )

    return {
        "ok": True,
        "total": len(items),
        "limit": limit,
        "results": items[:max(1, min(limit, 500))]
    }