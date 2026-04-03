from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

SUPPORTED_SOURCES = ["KBS", "MSN", "FMP", "VCI"]


class BatchRequest(BaseModel):
    symbols: list[str]


def get_price_from_sources(symbol: str):
    from vnstock import Vnstock

    errors = []
    code = (symbol or "").strip().upper()

    for source in SUPPORTED_SOURCES:
        try:
            stock = Vnstock().stock(symbol=code, source=source)
            df = stock.quote.history(
                start="2025-01-01",
                end=datetime.now().strftime("%Y-%m-%d"),
                interval="1D"
            )

            if df is not None and len(df) > 0:
                last_row = df.iloc[-1]

                for col in ["close", "Close", "c"]:
                    if col in df.columns:
                        price = float(last_row[col])
                        return {
                            "ok": True,
                            "symbol": code,
                            "currentPrice": price,
                            "priceDate": datetime.now().strftime("%Y-%m-%d"),
                            "source": source
                        }

                errors.append(f"{source}: no close column")
            else:
                errors.append(f"{source}: empty dataframe")

        except Exception as e:
            errors.append(f"{source}: {str(e)}")

    return {
        "ok": False,
        "symbol": code,
        "error": "All sources failed",
        "details": errors
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/price")
def get_price(symbol: str):
    return get_price_from_sources(symbol)


@app.post("/batch_prices")
def batch_prices(req: BatchRequest):
    results = []
    seen = set()

    for symbol in req.symbols:
        code = (symbol or "").strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        results.append(get_price_from_sources(code))

    success = sum(1 for x in results if x.get("ok"))

    return {
        "ok": True,
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "results": results
    }