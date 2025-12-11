#!/usr/bin/env python3
import base64
import csv
import math
import os
import sys
import time
import asyncio
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field

import aiohttp
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from rich.console import Console
from rich.table import Table

from src.config import Config, MarketState, YIELD_BUCKETS
from src.utils.opt_chain import generate_symbol_map, calculate_underlying_from_parity
from src.table_display import run_model

load_dotenv()
console = Console()

EVENT_TICKER = "KXTNOTED-25DEC12"
MIN_EDGE_PCT = 0.3
MIN_SPREAD_CENTS = 3
MAX_BID = 90
MIN_PROB_TO_BID = 5.0

API_KEY = os.getenv("KALSHI_API_KEY", "")
PRIVATE_KEY = os.getenv("KALSHI_PRIVATE_KEY", "")
REST_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def sign_request(private_key_str: str, timestamp: str, method: str, path: str) -> str:
    message = f"{timestamp}{method}{path}".encode('utf-8')
    private_key = serialization.load_pem_private_key(private_key_str.encode(), password=None)
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')


def create_headers(method: str, path: str) -> dict:
    timestamp = str(int(time.time() * 1000))
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": API_KEY,
        "KALSHI-ACCESS-SIGNATURE": sign_request(PRIVATE_KEY, timestamp, method, path),
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }


@dataclass
class Market:
    ticker: str
    yes_bid: int = 0
    yes_ask: int = 0


def parse_ticker_value(ticker: str) -> tuple[str, float, str]:
    suffix = ticker.split("-")[-1] if "-" in ticker else ticker
    match = re.match(r'^([BT])([\d.]+)$', suffix)
    if match:
        prefix, value = match.groups()
        v = float(value)
        if prefix == "B":
            return f"< {value}%", v, "below"
        return f"{value}%", v, "tail"
    return suffix, 0, "unknown"


async def fetch_kalshi_markets() -> dict:
    async with aiohttp.ClientSession() as session:
        path = f"/trade-api/v2/markets?event_ticker={EVENT_TICKER}&limit=200"
        headers = create_headers("GET", "/trade-api/v2/markets")
        async with session.get(f"{REST_BASE}/markets?event_ticker={EVENT_TICKER}&limit=200", headers=headers) as resp:
            data = await resp.json()
        
        markets = {}
        for m in data.get("markets", []):
            ticker = m.get("ticker", "")
            markets[ticker] = Market(ticker=ticker)
        
        for ticker in markets:
            path = f"/trade-api/v2/markets/{ticker}/orderbook"
            headers = create_headers("GET", path)
            async with session.get(f"{REST_BASE}/markets/{ticker}/orderbook", headers=headers) as resp:
                ob = await resp.json()
            
            orderbook = ob.get("orderbook", {})
            yes_bids = orderbook.get("yes", [])
            no_bids = orderbook.get("no", [])
            
            if yes_bids:
                markets[ticker].yes_bid = yes_bids[0][0]
            if no_bids:
                markets[ticker].yes_ask = 100 - no_bids[0][0]
        
        buckets = {}
        for ticker, market in markets.items():
            label, val, mtype = parse_ticker_value(ticker)
            if market.yes_bid and market.yes_ask:
                mid = (market.yes_bid + market.yes_ask) / 2
                buckets[val] = (mid, market, mtype)
        
        result = {}
        sorted_vals = sorted(buckets.keys())
        for i, val in enumerate(sorted_vals):
            mid, market, mtype = buckets[val]
            if mtype == "tail":
                if val == min(sorted_vals):
                    label = f"{val - 0.01:.2f}% or below".replace(".00%", "%")
                else:
                    label = f"{val + 0.01:.2f}% or above".replace(".00%", "%")
                result[label] = (mid, market)
            elif mtype == "below":
                label = f"{val - 0.01:.2f}% to {val + 0.01:.2f}%".replace(".00%", "%")
                result[label] = (mid, market)
        
        return result


def load_model_probs(csv_path: str = None, yield_10y: float = 4.20) -> dict:
    csv_path = csv_path or str(Path(__file__).parent / "data" / "TN_Dec10.csv")
    config = Config()
    
    last_line = open(csv_path).read().strip().split('\n')[-1]
    date_time_str = last_line.split('as of ')[1].replace(' CST', '').strip('"')
    config.reference_time = datetime.strptime(date_time_str, "%m-%d-%Y %I:%M%p").replace(tzinfo=ZoneInfo("America/Chicago"))
    
    state = MarketState()
    strikes = []
    
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            strike_str = row.get('Strike', '')
            if not strike_str or 'Downloaded' in strike_str:
                continue
            option_type = row.get('Type')
            if option_type not in ('Call', 'Put'):
                continue
            opt_char = 'C' if option_type == 'Call' else 'P'
            strike = float(strike_str[:-1])
            if strike not in strikes:
                strikes.append(strike)
            symbol = f"{config.contract_root}{opt_char}{strike:g}{config.exchange}"
            state.quotes[symbol] = {
                'BID_PRICE': float(row.get('Bid', 0) or 0),
                'ASK_PRICE': float(row.get('Ask', 0) or 0)
            }
    
    strikes = sorted(strikes)
    symbol_map = generate_symbol_map(config, strikes)
    state.underlying_price = calculate_underlying_from_parity(strikes, symbol_map, state, yield_adjustment_bp=1.5)
    state.yield_10y = yield_10y
    
    result = run_model(config, strikes, symbol_map, state, y10_yield=yield_10y)
    return result.bucket_probs if result else {}


def calculate_quote(prob: float) -> tuple[int, int]:
    fair = prob
    spread = max(fair * MIN_EDGE_PCT, MIN_SPREAD_CENTS)
    bid = int(fair - spread)
    ask = math.ceil(fair + spread)
    
    bid = max(1, min(bid, MAX_BID))
    ask = max(bid + 1, min(99, ask))
    
    if prob < MIN_PROB_TO_BID or bid >= fair:
        bid = 0
    if ask <= fair:
        ask = 0
    
    return bid, ask


def display_comparison(model_probs: dict, kalshi_data: dict):
    table = Table(title=f"Model vs Kalshi: {EVENT_TICKER}", show_header=True, header_style="bold cyan")
    table.add_column("Bucket", justify="left")
    table.add_column("Model", justify="right", style="yellow")
    table.add_column("Fair", justify="right")
    table.add_column("Bid", justify="right", style="green")
    table.add_column("Ask", justify="right", style="red")
    table.add_column("K.Bid", justify="right", style="white")
    table.add_column("K.Ask", justify="right", style="white")
    
    for label, _, _ in YIELD_BUCKETS:
        model_prob = model_probs.get(label, 0)
        kalshi_info = kalshi_data.get(label)
        
        bid, ask = calculate_quote(model_prob)
        bid_str = str(bid) if bid > 0 else "-"
        ask_str = str(ask) if ask > 0 else "-"
        
        if kalshi_info:
            _, market = kalshi_info
            k_bid = market.yes_bid or 0
            k_ask = market.yes_ask or 0
            table.add_row(
                label, f"{model_prob:.1f}%", f"{int(model_prob)}", bid_str, ask_str,
                str(k_bid) if k_bid else "-", str(k_ask) if k_ask else "-"
            )
        else:
            table.add_row(label, f"{model_prob:.1f}%", f"{int(model_prob)}", bid_str, ask_str, "-", "-")
    
    console.print(table)


async def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    yield_10y = float(sys.argv[2]) if len(sys.argv) > 2 else 4.20
    
    console.print("[dim]Loading model...[/dim]")
    model_probs = load_model_probs(csv_path, yield_10y)
    
    console.print("[dim]Fetching Kalshi...[/dim]")
    kalshi_data = await fetch_kalshi_markets()
    
    console.print()
    display_comparison(model_probs, kalshi_data)


if __name__ == "__main__":
    asyncio.run(main())
