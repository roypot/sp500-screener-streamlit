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
    Primary source: Wikipedia constituents table (live).
    Fetch with a browser-like User-Agent to reduce 403s, then parse HTML using lxml.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = SESSION.get(url, headers=UA_HEADERS, timeout=20)
        resp.raise_for_status()  # bubble up 403/404/etc
        tables = pd.read_html(resp.text, flavor="lxml")  # requires lxml installed
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
    Fallback sources if Wikipedia blocks or times out:
      - TradingView components page
      - TopForeignStocks S&P 500 components list
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
            has_ticker = "ticker" in cols_lower
            has_company = ("company" in cols_lower) or ("name" in cols_lower) or ("company name" in cols_lower)
            if has_ticker and has_company:
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
    """
    Download daily price history via yfinance and compute indicators:
    200-SMA, RSI(14), MACD(12,26,9), ADX(14), ROC(252).
    """
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return df

    # Indicators
    df["SMA200"] = ta.sma(df["Close"], length=SMA_LEN)
    df["RSI"]    = ta.rsi(df["Close"], length=RSI_LEN)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"]        = macd["MACD_12_26_9"]
        df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]

    adx = ta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=ADX_LEN)
    if adx is not None and not adx.empty:
        df["ADX"] = adx["ADX_14"]

    df["ROC"] = ta.roc(df["Close"], length=ROC_LEN)

    return df.dropna().copy()

# ------------------------
# Scoring helpers
# ------------------------
def percentile_value(series: pd.Series, x: float) -> float:
    """Fraction of peers <= x (0..1)."""
    s = pd.Series(series).dropna()
    if len(s) == 0 or pd.isna(x):
        return np.nan
    return float((s <= x).mean())

def score_low_good(x, peer):
    """Lower is better → higher score (0..10)."""
    p = percentile_value(peer, x)
    return 10 * (1 - p) if pd.notna(p) else np.nan

def score_high_good(x, peer):
    """Higher is better → higher score (0..10)."""
    p = percentile_value(peer, x)
    return 10 * p if pd.notna(p) else np.nan

def score_rsi(rsi):
    """10 at RSI=50; 0 at <=30 or >=70 (linear)."""
    if pd.isna(rsi):
        return np.nan
    return max(0.0, 10.0 * (1 - abs(rsi - 50) / 20))

def score_macd(macd, signal):
    """MACD positive & above signal → 10; above signal → 6; else 0."""
    if pd.isna(macd) or pd.isna(signal):
        return np.nan
    if macd > 0 and macd > signal:
        return 10.0
    if macd > signal:
        return 6.0
    return 0.0

def score_price_vs_200(close, sma200):
    """Premium vs 200-SMA: full 5 pts at +10% or more (0..5)."""
    if pd.isna(close) or pd.isna(sma200) or sma200 == 0:
        return np.nan
    prem = (close / sma200) - 1
    return 5.0 * float(np.clip(prem / 0.10, 0, 1))

def score_adx(adx):
    """Trend strength (ADX): >=25→10; 20–25→7; 15–20→3; else 0."""
    if pd.isna(adx):
        return np.nan
    if adx >= 25: return 10.0
    if adx >= 20: return 7.0
    if adx >= 15: return 3.0
    return 0.0

def score_roc(roc, peer):
    """12‑month ROC percentile → 0..10."""
    p = percentile_value(peer, roc)
    return 10 * p if pd.notna(p) else np.nan

# ------------------------
# UI (render controls first)
# ------------------------
src_banner = st.empty()
left, right = st.columns([1, 3])
with left:
    # Placeholder; real sector list appears after loading
    pick_sector = st.selectbox("Filter sector (optional)", ["All"])
    top_n       = st.slider("Show Top N", 10, 200, 50, step=10)
    run_btn     = st.button("Run Screening", type="primary")
with right:
    st.info("Full 100‑point scoring requires an **FMP API key** in Secrets. "
            "Without it, the app still computes **Technicals & Trend** (45 pts).")

# ------------------------
# Run workflow
# ------------------------
if run_btn:
    # Load constituents with graceful fallback
    with st.spinner("Loading S&P 500 constituents…"):
        meta = load_sp500_tickers()
        source_used = "Wikipedia"
        if meta.empty:
            meta = load_sp500_fallback()
            source_used = "Fallback source"

    if meta.empty:
        st.error("Could not load S&P 500 constituents from any source right now. Please retry.")
        st.stop()

    src_banner.info(f"Universe size: **{len(meta)}** • Source: **{source_used}**")

    # Now that we have data, refresh sector options
    sectors = ["All"] + sorted(meta["sector"].unique().tolist())
    pick_sector = st.selectbox("Filter sector (optional)", sectors)

    # Apply sector filter
    universe = meta.copy()
    if pick_sector != "All":
        universe = universe[universe["sector"] == pick_sector]

    symbols = universe["ticker"].tolist()

    # Fundamentals (if key present)
    try:
        fund = fmp_ratios(symbols)
    except RuntimeError as e:
        st.warning(str(e))
        fund = pd.DataFrame()
    have_fund = not fund.empty

    # Technicals: compute latest snapshot for each symbol
    tech_rows = []
    for sym in symbols:
        df = price_history(sym)
        if df.empty:
            continue
        last = df.iloc[-1]
        tech_rows.append({
            "ticker": sym,
            "close": last.get("Close"),
            "rsi": last.get("RSI"),
            "macd": last.get("MACD"),
            "macd_signal": last.get("MACD_SIGNAL"),
            "sma200": last.get("SMA200"),
            "adx": last.get("ADX"),
            "roc": last.get("ROC"),
        })
    tech = pd.DataFrame(tech_rows)

    # Merge
    df = universe.merge(tech, on="ticker", how="left")
    if have_fund:
        df = df.merge(fund, on="ticker", how="left")

    # Peer vectors for percentile scoring
    peers = {"roc": df["roc"]}
    if have_fund:
        peers.update({
            "ps_ratio": df["ps_ratio"],
            "peg_ratio": df["peg_ratio"],
            "fcf_yield": df["fcf_yield"],
            "de_ratio": df["de_ratio"],
            "gross_margin": df["gross_margin"],
            "revenue_growth": df["revenue_growth"],
        })

    # Scores
    if have_fund:
        df["score_ps"]   = df["ps_ratio"].apply(lambda x: score_low_good(x, peers["ps_ratio"]))
        df["score_peg"]  = df["peg_ratio"].apply(lambda x: score_low_good(x, peers["peg_ratio"]))
        df["score_fcfy"] = df["fcf_yield"].apply(lambda x: score_high_good(x, peers["fcf_yield"]))
        df["score_de"]   = df["de_ratio"].apply(lambda x: score_low_good(x, peers["de_ratio"]))
        df["score_gm"]   = df["gross_margin"].apply(lambda x: score_high_good(x, peers["gross_margin"]))
        # Revenue Growth worth 5 points (half-weight):
        df["score_rev"]  = df["revenue_growth"].apply(lambda x: 0.5 * score_high_good(x, peers["revenue_growth"]))
    else:
        for c in ["score_ps","score_peg","score_fcfy","score_de","score_gm","score_rev"]:
            df[c] = np.nan

    df["score_rsi"]   = df["rsi"].apply(score_rsi)
    df["score_macd"]  = df.apply(lambda r: score_macd(r["macd"], r["macd_signal"]), axis=1)
    df["score_pv200"] = df.apply(lambda r: score_price_vs_200(r["close"], r["sma200"]), axis=1)
    df["score_adx"]   = df["adx"].apply(score_adx)
    df["score_roc"]   = df["roc"].apply(lambda x: score_roc(x, peers["roc"]))

    # Category totals
    df["Valuation"] = df[["score_ps","score_peg","score_fcfy"]].sum(axis=1, min_count=1)
    df["Quality"]   = df[["score_de","score_gm","score_rev"]].sum(axis=1, min_count=1)
    df["Momentum"]  = df[["score_rsi","score_macd","score_pv200"]].sum(axis=1, min_count=1)
    df["Trend"]     = df[["score_adx","score_roc"]].sum(axis=1, min_count=1)

    # Total (works even if fundamentals are missing)
    df["TotalScore"] = df[["Valuation","Quality","Momentum","Trend"]].sum(axis=1, min_count=1)

    out_cols = [
        "ticker","company","sector","TotalScore","Valuation","Quality","Momentum","Trend",
        "ps_ratio","peg_ratio","fcf_yield","de_ratio","gross_margin","revenue_growth",
        "rsi","macd","macd_signal","adx","roc","close"
    ]
    view = df[out_cols].sort_values("TotalScore", ascending=False).head(top_n).reset_index(drop=True)

    st.subheader("Top Ranked")
    st.dataframe(view, use_container_width=True)

    # Download
    st.download_button(
        "Download CSV",
        view.to_csv(index=False).encode(),
        file_name="sp500_ratings.csv",
        mime="text/csv"
    )

    # Drill‑down charts
    sel = st.selectbox("Show charts for:", view["ticker"])
    if sel:
        hist = price_history(sel, period="3y")
        if not hist.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{sel} – Price with 200‑SMA**")
                st.line_chart(hist[["Close","SMA200"]])
                st.caption("Prices via Yahoo Finance (`yfinance`). Indicators via `pandas_ta`.")
            with c2:
                st.markdown("**RSI & MACD**")
                st.line_chart(hist["RSI"])
                macd_plot = hist[["MACD","MACD_SIGNAL"]].dropna()
                if not macd_plot.empty:
                    st.line_chart(macd_plot)
            c3, _ = st.columns([1,1])
            with c3:
                st.markdown("**ADX & ROC**")
                st.line_chart(hist[["ADX","ROC"]])
