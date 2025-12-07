# S&P 500 Stock Rater (Streamlit)

Ranks S&P 500 companies on a 100‑point model:

- **Valuation (30)**: P/S, PEG, Free Cash Flow Yield
- **Financial Quality (25)**: Debt/Equity, Gross Margin, Revenue Growth
- **Technical Momentum (25)**: RSI, MACD, Price vs 200‑SMA
- **Trend Strength (20)**: ADX, 12‑month ROC

## Run locally
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
