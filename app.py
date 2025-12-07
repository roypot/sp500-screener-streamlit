# app.py
import time, requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import pandas_ta as ta

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="S&P 500 Stock Rater", page_icon="📈", layout="wide")
st.title("S&P 500 Stock Rater (100-point model)")

# ------------------------
# Settings & constants
# ------------------------
FMP_KEY = st.secrets.get("FMP_API_KEY", "")
SESSION = requests.Session()

RSI_LEN = 14
ADX_LEN = 14
ROC_LEN = 252   # ~12 months of daily changes
SMA_LEN = 200

# ------------------------
# Helpers
# ------------------------
@st.cache_data(ttl=6*60*60)
def load_sp500_tickers() -> pd.DataFrame:
    """
    Fetch S&P 500 constituents from Wikipedia.
    We request the page with a browser-like User-Agent to avoid HTTP 403,
    then parse the HTML text with lxml.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = SESSION.get(url, headers=headers, timeout=30)
    resp.raise_for_status()  # will raise HTTPError if 403/404/...
    tables = pd.read_html(resp.text, flavor="lxml")  # requires lxml
    df = tables[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "company", "GICS Sector": "sector"})
    # Yahoo uses '-' instead of '.' for some tickers (e.g., BRK.B -> BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "company", "sector"]]

@st.cache_data(ttl=24*60*60, show_spinner=False)
def fmp_ratios(symbols: list[str]) -> pd.DataFrame:
    """
    Pull the fundamentals we need from FMP 'stable' endpoints:
      - /stable/ratios?symbol=...
      - /stable/key-metrics?symbol=...
    Includes clearer 403 error messages and gentler pacing for free-tier keys.
    """
    if not FMP_KEY:
        return pd.DataFrame()

    def fetch(endpoint, params):
        url = f"https://financialmodelingprep.com/stable/{endpoint}"
        params = params | {"apikey": FMP_KEY}
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 403:
            raise RuntimeError(
                f"FMP returned 403 for '{endpoint}'. "
                "Check your API key (Secrets) and free-tier rate limits. "
                "Try reducing Top N or filtering by sector."
            )
        r.raise_for_status()
        return r.json()

    rows = []
    for sym in symbols:
        try:
            ratios = fetch("ratios", {"symbol": sym, "limit": 1, "period": "FY"})
            km     = fetch("key-metrics", {"symbol": sym, "limit": 1, "period": "FY"})
            r = ratios[0] if ratios else {}
            k = km[0] if km else {}
            rows.append({
                "ticker": sym,
                # Valuation
                "ps_ratio":     r.get("priceToSalesRatio", np.nan) or r.get("priceToSalesRatioTTM", np.nan),
                "peg_ratio":    r.get("pegRatio", np.nan) or r.get("pegRatioTTM", np.nan),
                "fcf_yield":    k.get("freeCashFlowYield", np.nan) or k.get("freeCashFlowYieldTTM", np.nan),
                # Quality
                "de_ratio":     r.get("debtToEquity", np.nan) or r.get("debtToEquityTTM", np.nan),
                "gross_margin": r.get("grossProfitMargin", np.nan) or r.get("grossProfitMarginTTM", np.nan),
                "revenue_growth": r.get("revenueGrowth", np.nan) or r.get("revenueGrowthTTM", np.nan),
            })
        except Exception:
            rows.append({"ticker": sym})  # keep going if one symbol fails
        time.sleep(0.2)  # be polite: reduce chance of hitting free-tier 403s

    return pd.DataFrame(rows)

@st.cache_data(ttl=12*60*60, show_spinner=False)
def price_history(symbol: str, period="3y") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return df
    # Indicators
    df["SMA200"] = ta.sma(df["Close"], length=SMA_LEN)
    df["RSI"]    = ta.rsi(df["Close"], length=RSI_LEN)
    macd         = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"]        = macd["MACD_12_26_9"]
        df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]
