from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd

app = FastAPI()

SUPPORTED_SOURCES = ["KBS", "MSN", "FMP", "VCI"]


class BatchRequest(BaseModel):
    symbols: list[str]


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def build_indicators(df: pd.DataFrame) -> dict:
    df = df.copy()

    close_col = None
    volume_col = None
    high_col = None

    for c in ["close", "Close", "c"]:
      if c in df.columns:
        close_col = c
        break

    for c in ["volume", "Volume", "v"]:
      if c in df.columns:
        volume_col = c
        break

    for c in ["high", "High", "h"]:
      if c in df.columns:
        high_col = c
        break

    if not close_col:
        raise ValueError("No close column found")

    df["close_x"] = pd.to_numeric(df[close_col], errors="coerce")
    df["volume_x"] = pd.to_numeric(df[volume_col], errors="coerce") if volume_col else 0
    df["high_x"] = pd.to_numeric(df[high_col], errors="coerce") if high_col else df["close_x"]

    df["ma20"] = df["close_x"].rolling(20).mean()
    df["ma50"] = df["close_x"].rolling(50).mean()
    df["ma200"] = df["close_x"].rolling(200).mean()
    df["vol_ma20"] = df["volume_x"].rolling(20).mean()
    df["rsi14"] = calc_rsi(df["close_x"], 14)

    df["high_20_prev"] = df["high_x"].rolling(20).max().shift(1)

    last = df.iloc[-1]

    return {
        "currentPrice": float(last["close_x"]) if pd.notna(last["close_x"]) else 0,
        "ma20": float(last["ma20"]) if pd.notna(last["ma20"]) else 0,
        "ma50": float(last["ma50"]) if pd.notna(last["ma50"]) else 0,
        "ma200": float(last["ma200"]) if pd.notna(last["ma200"]) else 0,
        "rsi14": float(last["rsi14"]) if pd.notna(last["rsi14"]) else 0,
        "volume": float(last["volume_x"]) if pd.notna(last["volume_x"]) else 0,
        "volMa20": float(last["vol_ma20"]) if pd.notna(last["vol_ma20"]) else 0,
        "breakoutLevel": float(last["high_20_prev"]) if pd.notna(last["high_20_prev"]) else 0
    }


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
    code = (symbol or "").strip().upper()
    errors = []

    for source in SUPPORTED_SOURCES:
        try:
            df = fetch_stock_df(code, source)

            if df is None or len(df) < 30:
                errors.append(f"{source}: insufficient data")
                continue

            indicators = build_indicators(df)

            return {
                "ok": True,
                "symbol": code,
                "source": source,
                "priceDate": datetime.now().strftime("%Y-%m-%d"),
                **indicators
            }

        except Exception as e:
            errors.append(f"{source}: {str(e)}")

    return {
        "ok": False,
        "symbol": code,
        "error": "All sources failed",
        "details": errors
    }


@app.get("/")
def root():
    return {"ok": True, "service": "vn-signal-bridge"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/signal")
def signal(symbol: str):
    return get_signal_payload(symbol)


@app.post("/batch_signals")
def batch_signals(req: BatchRequest):
    results = []
    seen = set()

    for symbol in req.symbols:
        code = (symbol or "").strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        results.append(get_signal_payload(code))

    success = sum(1 for x in results if x.get("ok"))

    return {
        "ok": True,
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "results": results
    }
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

    "CRC": ("Retail", "Midcap")
}

def infer_sector_and_group(symbol: str):
    code = (symbol or "").strip().upper()
    if code in SECTOR_MAP:
        return SECTOR_MAP[code]
    return ("Other", "General")

def classify_liquidity(vol_ma20: float):
    if vol_ma20 >= 1000000:
        return "High"
    if vol_ma20 >= 300000:
        return "Medium"
    return "Low"


@app.post("/batch_universe")
def batch_universe(req: BatchRequest):
    results = []
    seen = set()

    for symbol in req.symbols:
        code = (symbol or "").strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)

        payload = get_signal_payload(code)

        if not payload.get("ok"):
            sector, group_tag = infer_sector_and_group(code)
            results.append({
                "ok": False,
                "symbol": code,
                "sector": sector,
                "groupTag": group_tag,
                "liquidityTier": "Low",
                "active": False,
                "error": payload.get("error", "No data")
            })
            continue

        sector, group_tag = infer_sector_and_group(code)
        vol_ma20 = float(payload.get("volMa20", 0) or 0)
        liquidity_tier = classify_liquidity(vol_ma20)
        active = liquidity_tier in ["High", "Medium"]

        results.append({
            "ok": True,
            "symbol": code,
            "sector": sector,
            "groupTag": group_tag,
            "liquidityTier": liquidity_tier,
            "active": active,
            "currentPrice": payload.get("currentPrice", 0),
            "volume": payload.get("volume", 0),
            "volMa20": payload.get("volMa20", 0),
            "priceDate": payload.get("priceDate", "")
        })

    success = sum(1 for x in results if x.get("ok"))

    return {
        "ok": True,
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "results": results
    }