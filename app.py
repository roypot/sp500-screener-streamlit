# app.py
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import pandas_ta as ta
from typing import List

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
ROC_LEN = 252   # ~12 months
SMA_LEN = 200

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# ------------------------
# Data loaders
# ------------------------
@st.cache_data(ttl=6*60*60)
def load_sp500_tickers() -> pd.DataFrame:
    """
    Primary source: Wikipedia constituents table.
    Fetch with a browser-like User-Agent to reduce 403s, then parse the HTML using lxml.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = SESSION.get(url, headers=UA_HEADERS, timeout=20)
        resp.raise_for_status()  # bubble up 403/404/etc
        tables = pd.read_html(resp.text, flavor="lxml")  # requires lxml
        df = tables[0]
        df = df.rename(columns={"Symbol": "ticker", "Security": "company", "GICS Sector": "sector"})
        # Yahoo tickers use '-' instead of '.' (e.g., BRK.B -> BRK-B)
        df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
        return df[["ticker", "company", "sector"]]
    except Exception as e:
        # Return empty so the caller can switch to the fallback
        st.warning(f"Wikipedia source unavailable ({type(e).__name__}): falling back.")
        return pd.DataFrame(columns=["ticker", "company", "sector"])

@st.cache_data(ttl=6*60*60)
def load_sp500_fallback() -> pd.DataFrame:
    """
    Fallback sources:
      - TradingView components page (may or may not render tables in static HTML)
      - TopForeignStocks S&P 500 components list (frequently updated static table)
    """
    # Option A: TradingView components
    try:
        url_tv = "https://www.tradingview.com/symbols/SPX/components/"
        resp = SESSION.get(url_tv, headers=UA_HEADERS, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(resp.text, flavor="lxml")
        if len(tables) > 0:
            df = tables[0]
            # Normalize common column names across variants
            if "Symbol" in df.columns:
                df = df.rename(columns={"Symbol": "ticker"})
            elif "Ticker" in df.columns:
                df = df.rename(columns={"Ticker": "ticker"})
            if "Company" in df.columns:
                df = df.rename(columns={"Company": "company"})
            elif "Name" in df.columns:
                df = df.rename(columns={"Name": "company"})
            # Sector if present; otherwise "Unknown"
            df["sector"] = df["Sector"] if "Sector" in df.columns else "Unknown"
            df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
            if set(["ticker", "company", "sector"]).issubset(df.columns):
                return df[["ticker", "company", "sector"]].dropna()
    except Exception:
        pass  # fall through to secondary source

    # Option B: TopForeignStocks
    try:
        url_tfs = "https://topforeignstocks.com/indices/components-of-the-sp-500-index/"
        resp = SESSION.get(url_tfs, headers=UA_HEADERS, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(resp.text, flavor="lxml")
        # Look for a table with Ticker + Company/Name + Sector/Industry
        for t in tables:
            cols_lower = {str(c).lower(): c for c in t.columns}
            if "ticker" in cols_lower and (
                "name" in cols_lower or "company name" in cols_lower or "company" in cols_lower
            ):
                company_col = cols_lower.get("company") or cols_lower.get("name") or cols_lower.get("company name")
                df = t.rename(columns={cols_lower["ticker"]: "ticker", company_col: "company"})
                sector_col = cols_lower.get("sector") or cols_lower.get("industry")
                if sector_col:
                    df = df.rename(columns={sector_col: "sector"})
                else:
                    df["sector"] = "Unknown"
                df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
                if set(["ticker", "company", "sector"]).issubset(df.columns):
                    return df[["ticker", "company", "sector"]].dropna()
    except Exception:
        pass

    return pd.DataFrame(columns=["ticker", "company", "sector"])

@st.cache_data(ttl=24*60*60, show_spinner=False)
def fmp_ratios(symbols: List[str]) -> pd.DataFrame:
    """
    Fundamentals from FMP 'stable' endpoints (Ratios + Key Metrics).
    Clearer 403 handling and gentle pacing to respect free-tier rate limits.
    """
    if not FMP_KEY:
        return pd.DataFrame()

    def fetch(endpoint: str, params: dict) -> list:
        url = f"https://financialmodelingprep.com/stable/{endpoint}"
        params = params | {"apikey": FMP_KEY}
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 403:
            raise RuntimeError(
                f"FMP returned 403 for '{endpoint}'. "
                "Check your API key (Secrets) and rate limits. "
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
                # Valuation (30)
                "ps_ratio":     r.get("priceToSalesRatio", np.nan) or r.get("priceToSalesRatioTTM", np.nan),
                "peg_ratio":    r.get("pegRatio", np.nan) or r.get("pegRatioTTM", np.nan),
                "fcf_yield":    k.get("freeCashFlowYield", np.nan) or k.get("freeCashFlowYieldTTM", np.nan),
                # Financial Quality (25)
                "de_ratio":     r.get("debtToEquity", np.nan) or r.get("debtToEquityTTM", np.nan),
                "gross_margin": r.get("grossProfitMargin", np.nan) or r.get("grossProfitMarginTTM", np.nan),
                "revenue_growth": r.get("revenueGrowth", np.nan) or r.get("revenueGrowthTTM", np.nan),
            })
        except Exception:
            # Keep going even if one symbol fails
            rows.append({"ticker": sym})
        time.sleep(0.2)  # polite pacing for free tier

    return pd.DataFrame(rows)

@st.cache_data(ttl=12*60*60, show_spinner=False)
def price_history(symbol: str, period="3y") -> pd.DataFrame:
