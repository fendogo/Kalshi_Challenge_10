# 10Y Yield Profile Predictor

Extracts implied probability distributions from TN futures options to predict 10Y Treasury yield ranges.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

### snapshot.py

Generate probability distribution from CSV option data:

```bash
python snapshot.py                           # uses data/TN_Dec10.csv
python snapshot.py data/TN_Dec9.csv          # custom CSV
```

Outputs:
- Options chain table with IV
- Yield bucket probabilities
- `vol_fit.png` - SVI volatility surface
- `pdf_fit.png` - implied probability density

### kalshi_compare.py

Compare model probabilities to live Kalshi markets:

```bash
python kalshi_compare.py                     # uses default CSV
python kalshi_compare.py data/TN_Dec10.csv   # custom CSV
```

Requires Kalshi API credentials in parent directory `.env`:
```
KALSHI_API_KEY=your_key
KALSHI_PRIVATE_KEY=your_private_key
```

### run_stream.py

Live streaming display with Schwab API:

```bash
python run_stream.py
```

Requires:
- `token.json` - Schwab auth token
- `.env` with `API_KEY` and `APP_SECRET`

## CSV Format

Option data CSV should have columns:
- `Strike` - e.g., "115.00C" or "115.00P"
- `Type` - "Call" or "Put"
- `Bid` - bid price in decimal
- `Ask` - ask price in decimal

Last line should contain timestamp: `"Downloaded from ... as of MM-DD-YYYY HH:MMam/pm CST"`
