#!/usr/bin/env python3
import csv
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from kalshi.client import KalshiClient
from kalshi.monitor import KalshiWebSocket
from src.config import Config, MarketState
from src.utils.math import price_to_yield
from src.utils.opt_chain import generate_symbol_map, calculate_underlying_from_parity
from src.table_display import run_model

console = Console()

EVENT_TICKER = "KXTNOTED-25DEC31"
SPREAD = 2


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


async def fetch_kalshi_markets() -> dict:
    async with KalshiClient() as client:
        tickers = await client.load_markets(event_ticker=EVENT_TICKER)
        for ticker in tickers:
            ob = await client.rest.get_orderbook(ticker)
            KalshiWebSocket.update_orderbook_from_data(client.state, ticker, ob.get("orderbook", {}))
        event = client.state.events.get(EVENT_TICKER)
        if not event:
            return {}
        return {b.label: (b.prob, b.cum_market) for b in event.get_buckets()}


def display_comparison(model_probs: dict, kalshi_data: dict):
    table = Table(title=f"Model vs Kalshi: {EVENT_TICKER}", show_header=True, header_style="bold cyan")
    table.add_column("Bucket", justify="left")
    table.add_column("Model", justify="right", style="yellow")
    table.add_column("Fair", justify="right")
    table.add_column("Bid", justify="right", style="green")
    table.add_column("Ask", justify="right", style="red")
    table.add_column("K.Bid", justify="right", style="dim green")
    table.add_column("K.Ask", justify="right", style="dim red")
    table.add_column("Edge", justify="right")
    
    buckets = [
        "3.92% or below", "3.93% to 3.95%", "3.96% to 3.98%", "3.99% to 4.01%",
        "4.02% to 4.04%", "4.05% to 4.07%", "4.08% to 4.1%", "4.11% to 4.13%",
        "4.14% to 4.16%", "4.17% to 4.19%", "4.2% to 4.22%", "4.23% to 4.25%",
        "4.26% to 4.28%", "4.29% to 4.31%", "4.32% or above",
    ]
    
    for label in buckets:
        model_prob = model_probs.get(label, 0)
        kalshi_info = kalshi_data.get(label)
        
        fair = int(model_prob)
        bid = max(1, fair - SPREAD)
        ask = min(99, fair + SPREAD)
        
        if kalshi_info:
            _, market = kalshi_info
            k_bid = market.yes_bid or 0
            k_ask = market.yes_ask or 0
            
            edge = ""
            if k_ask and fair > k_ask:
                edge = f"[green]+{fair - k_ask}[/green]"
            elif k_bid and fair < k_bid:
                edge = f"[red]{fair - k_bid}[/red]"
            
            table.add_row(
                label, f"{model_prob:.1f}%", str(fair), str(bid), str(ask),
                str(k_bid) if k_bid else "-", str(k_ask) if k_ask else "-", edge
            )
        else:
            table.add_row(label, f"{model_prob:.1f}%", str(fair), str(bid), str(ask), "-", "-", "")
    
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
