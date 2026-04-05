from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any
import math
import re
import traceback
import pandas as pd

app = FastAPI(title="vn-signal-bridge", version="3.0.1")

SUPPORTED_SOURCES = ["KBS", "MSN", "VCI"]
MAX_WORKERS = 8
CACHE_TTL_SECONDS = 900  # 15 minutes


class BatchRequest(BaseModel):
    symbols: list[str]


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
    "KBC": ("IndustrialZone", "Midcap"),
    "SZC": ("IndustrialZone", "Midcap"),

    "HPG": ("Steel", "VN30"),
    "HSG": ("Steel", "Midcap"),
    "NKG": ("Steel", "Midcap"),
    "GDA": ("Materials", "Midcap"),

    "MWG": ("Retail", "VN30"),
    "PNJ": ("Retail", "LargeCap"),
    "FRT": ("Retail", "Midcap"),
    "DGW": ("Retail", "Midcap"),
    "CRC": ("Retail", "Midcap"),
    "MSN": ("Consumer", "VN30"),
    "SAB": ("Consumer", "LargeCap"),
    "VNM": ("Consumer", "VN30"),

    "GAS": ("Energy", "VN30"),
    "POW": ("Energy", "VN30"),
    "PVD": ("OilGas", "Midcap"),
    "PVS": ("OilGas", "Midcap"),
    "BSR": ("OilGas", "Midcap"),
    "OIL": ("OilGas", "Midcap"),
    "PLX": ("OilGas", "LargeCap"),

    "GMD": ("Logistics", "LargeCap"),
    "HAH": ("Logistics", "Midcap"),
    "VSC": ("Logistics", "Midcap"),
    "PHP": ("Logistics", "Midcap"),
    "SCS": ("Logistics", "Midcap"),

    "DGC": ("Chemicals", "LargeCap"),
    "DPM": ("Chemicals", "Midcap"),
    "DCM": ("Chemicals", "Midcap"),
    "CSV": ("Chemicals", "Midcap"),
    "LAS": ("Chemicals", "Midcap"),

    "TCM": ("Textile", "Midcap"),
    "MSH": ("Textile", "Midcap"),
    "STK": ("Textile", "Midcap"),

    "VHC": ("Seafood", "Midcap"),
    "ANV": ("Seafood", "Midcap"),
    "FMC": ("Seafood", "Midcap"),
    "DBC": ("Agriculture", "Midcap"),
    "BAF": ("Agriculture", "Midcap"),

    "CTD": ("Construction", "Midcap"),
    "HBC": ("Construction", "Midcap"),
    "FCN": ("Construction", "Midcap"),
    "HHV": ("Infrastructure", "Midcap"),
    "C4G": ("Infrastructure", "Midcap"),
    "LCG": ("Infrastructure", "Midcap"),

    "FPT": ("Technology", "VN30"),
    "CMG": ("Technology", "Midcap"),
    "ELC": ("Technology", "Midcap"),

    "REE": ("Utilities", "LargeCap"),
    "GEG": ("Utilities", "Midcap"),
    "NT2": ("Utilities", "Midcap"),
    "QTP": ("Utilities", "Midcap"),
    "PPC": ("Utilities", "Midcap"),

    "HVN": ("Aviation", "Midcap"),
    "VJC": ("Aviation", "LargeCap"),
    "AST": ("Tourism", "Midcap"),
    "SKG": ("Tourism", "Midcap"),
}


DEFAULT_UNIVERSE = [
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "HDB", "STB", "SHB", "EIB",
    "SSI", "VND", "VIX", "HCM", "SHS", "MBS", "BSI", "FTS", "AGR",
    "VIC", "VHM", "NVL", "PDR", "KDH", "NLG", "DXG", "DIG", "CEO", "IDC", "KBC", "SZC",
    "HPG", "HSG", "NKG", "GDA",
    "MWG", "PNJ", "FRT", "DGW", "CRC", "MSN", "SAB", "VNM",
    "GAS", "POW", "PVD", "PVS", "BSR", "OIL", "PLX",
    "GMD", "HAH", "VSC", "PHP", "SCS",
    "DGC", "DPM", "DCM", "CSV", "LAS",
    "TCM", "MSH", "STK",
    "VHC", "ANV", "FMC", "DBC", "BAF",
    "CTD", "HBC", "FCN", "HHV", "C4G", "LCG",
    "FPT", "CMG", "ELC",
    "REE", "GEG", "NT2", "QTP", "PPC",
    "HVN", "VJC", "AST", "SKG",
]

RANK_ORDER = {
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1
}


_quote_cache: dict[str, dict[str, Any]] = {}
_cache_expiry: dict[str, float] = {}
_cache_lock = Lock()


def now_ts() -> float:
    return datetime.utcnow().timestamp()


def cache_get(key: str):
    with _cache_lock:
        exp = _cache_expiry.get(key)
        if not exp or exp < now_ts():
            _quote_cache.pop(key, None)
            _cache_expiry.pop(key, None)
            return None
        return _quote_cache.get(key)


def cache_set(key: str, value: dict[str, Any], ttl_seconds: int = CACHE_TTL_SECONDS):
    with _cache_lock:
        _quote_cache[key] = value
        _cache_expiry[key] = now_ts() + ttl_seconds


def normalize_symbol(symbol: str) -> str:
    code = (symbol or "").strip().upper()
    code = code.strip('"').strip("'").strip("`")
    code = re.sub(r"[^A-Z0-9]", "", code)
    return code


def unique_symbols(symbols: list[str]) -> list[str]:
    seen = set()
    out = []
    for symbol in symbols:
        code = normalize_symbol(symbol)
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


def rank_at_least(actual: str, minimum: str) -> bool:
    actual_score = RANK_ORDER.get((actual or "D").upper(), 1)
    minimum_score = RANK_ORDER.get((minimum or "D").upper(), 1)
    return actual_score >= minimum_score


def infer_sector_and_group(symbol: str):
    code = normalize_symbol(symbol)
    return SECTOR_MAP.get(code, ("Other", "General"))


def classify_liquidity(vol_ma20: float):
    if vol_ma20 >= 1_000_000:
        return "High"
    if vol_ma20 >= 300_000:
        return "Medium"
    return "Low"


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

    if price > ma20 > 0:
        score += 15
    if ma20 > ma50 > 0:
        score += 20
    if ma50 > ma200 > 0:
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


def passes_top_pick_filter(
    item: dict,
    min_score: float = 55,
    min_setup_rank: str = "B",
    active_only: bool = True,
    sector: str | None = None
) -> bool:
    if not item.get("ok"):
        return False

    if active_only and not item.get("active", False):
        return False

    if safe_float(item.get("signalScore", 0)) < min_score:
        return False

    if not rank_at_least(item.get("setupRank", "D"), min_setup_rank):
        return False

    trend_stage = str(item.get("trendStage", "Unknown"))
    if trend_stage not in ["Uptrend", "StrongUptrend"]:
        return False

    if sector and str(item.get("sector", "")).lower() != sector.lower():
        return False

    return True


def fetch_stock_df(symbol: str, source: str):
    from vnstock import Vnstock

    stock = Vnstock().stock(symbol=symbol, source=source)
    start_date = (datetime.now() - timedelta(days=420)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    return stock.quote.history(
        start=start_date,
        end=end_date,
        interval="1D"
    )


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


def scan_universe(symbols: list[str]) -> list[dict]:
    codes = unique_symbols(symbols)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(build_universe_item, code): code for code in codes}

        for future in as_completed(futures):
            code = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append({
                    "ok": False,
                    "symbol": code,
                    "sector": "Other",
                    "groupTag": "General",
                    "liquidityTier": "Low",
                    "active": False,
                    "trendStage": "Unknown",
                    "setupRank": "D",
                    "signalScore": 0,
                    "error": str(e),
                    "details": [traceback.format_exc()]
                })

    results.sort(
        key=lambda x: (
            0 if x.get("ok") else 1,
            -safe_float(x.get("signalScore", 0)),
            x.get("symbol", "")
        )
    )
    return results


@app.get("/")
def root():
    return {"ok": True, "service": "vn-signal-bridge", "version": "3.0.1"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "supportedSources": SUPPORTED_SOURCES,
        "cacheItems": len(_quote_cache),
        "universeSize": len(DEFAULT_UNIVERSE),
        "version": "3.0.1"
    }


@app.get("/signal")
def signal(symbol: str):
    try:
        return get_signal_payload(symbol)
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/batch_signals")
def batch_signals(req: BatchRequest):
    try:
        symbols = unique_symbols(req.symbols)
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(get_signal_payload, code): code for code in symbols}

            for future in as_completed(futures):
                code = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({
                        "ok": False,
                        "symbol": code,
                        "error": str(e),
                        "details": [traceback.format_exc()]
                    })

        results.sort(key=lambda x: x.get("symbol", ""))
        success = sum(1 for x in results if x.get("ok"))

        return {
            "ok": True,
            "total": len(results),
            "success": success,
            "failed": len(results) - success,
            "results": results
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/batch_universe")
def batch_universe(req: BatchRequest):
    try:
        symbols = unique_symbols(req.symbols)
        results = scan_universe(symbols)

        success = sum(1 for x in results if x.get("ok"))

        return {
            "ok": True,
            "total": len(results),
            "success": success,
            "failed": len(results) - success,
            "results": results
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/universe_auto")
def universe_auto(
    limit: int = 300,
    active_only: bool = False,
    sector: str | None = None
):
    try:
        universe = DEFAULT_UNIVERSE[:max(1, min(limit, len(DEFAULT_UNIVERSE)))]
        results = scan_universe(universe)

        if active_only:
            results = [x for x in results if x.get("active")]

        if sector:
            results = [
                x for x in results
                if str(x.get("sector", "")).lower() == sector.lower()
            ]

        success = sum(1 for x in results if x.get("ok"))

        return {
            "ok": True,
            "scanSize": len(universe),
            "total": len(results),
            "success": success,
            "failed": len(results) - success,
            "results": results
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/top_picks")
def top_picks(
    limit: int = 20,
    min_score: float = 55,
    min_setup_rank: str = "B",
    active_only: bool = True,
    sector: str | None = None
):
    try:
        results = scan_universe(DEFAULT_UNIVERSE)

        filtered = []
        for item in results:
            try:
                if passes_top_pick_filter(
                    item,
                    min_score=min_score,
                    min_setup_rank=min_setup_rank,
                    active_only=active_only,
                    sector=sector
                ):
                    filtered.append(item)
            except Exception:
                continue

        filtered.sort(
            key=lambda x: (
                -safe_float(x.get("signalScore", 0)),
                -safe_float(x.get("volumeRatio", 0)),
                x.get("symbol", "")
            )
        )

        top_results = filtered[:max(1, min(limit, 100))]

        return {
            "ok": True,
            "universeSize": len(DEFAULT_UNIVERSE),
            "matched": len(filtered),
            "returned": len(top_results),
            "filters": {
                "min_score": min_score,
                "min_setup_rank": min_setup_rank,
                "active_only": active_only,
                "sector": sector
            },
            "results": top_results
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }