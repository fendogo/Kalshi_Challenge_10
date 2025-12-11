# 10Y Yield Profile Predictor

## Dependencies

```bash
pip install -r requirements.txt
```

## snapshot.py

Generates implied probability distribution from TN futures options CSV data.

```bash
python snapshot.py                    # default: data/TN_Dec10.csv
python snapshot.py data/TN_Dec9.csv   # custom CSV
```

**Outputs:**
- Options chain table (strike, yield, IV, bid/ask)
- 10Y yield bucket probabilities
- `vol_fit.png` - SVI volatility surface fit
- `pdf_fit.png` - implied probability density

## kalshi_compare.py

Compares model probabilities to live Kalshi orderbook and generates suggested bid/ask prices.

```bash
python kalshi_compare.py
```

**Outputs:**
- Model probability per yield bucket
- Fair value, suggested bid/ask
- Current Kalshi bid/ask for comparison

Uses Kalshi REST API. Requires env vars (via `.env` in parent directory):
- `KALSHI_API_KEY`
- `KALSHI_PRIVATE_KEY`
