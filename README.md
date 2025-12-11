# 10Y Yield Profile Predictor

## Dependencies

```bash
pip install -r requirements.txt
```
## There are two ways to run the model: snapshot.py which runs the model on barchart.com CSV data, and run_stream.py for streaming the data from Schwab.

## snapshot.py requires no credentials and is given for the purpose of reproducibility. The final predictions are based on the free public barchart.com data.

## CSV source (https://www.barchart.com/futures/quotes/TNH26/options/BV2Z25?futuresOptionsView=merged)

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

Requires `.env` with:
```
KALSHI_API_KEY=your_key
KALSHI_PRIVATE_KEY=your_private_key
```
