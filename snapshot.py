#!/usr/bin/env python3
import csv
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rich.console import Console

from src.config import Config, MarketState
from src.utils.math import price_to_yield
from src.utils.opt_chain import generate_symbol_map, calculate_underlying_from_parity
from src.table_display import run_model, build_renderables, render_vol_fit, render_pdf

console = Console()


def parse_timestamp(csv_path: str) -> datetime:
    last_line = open(csv_path).read().strip().split('\n')[-1]
    date_time_str = last_line.split('as of ')[1].replace(' CST', '').strip('"')
    return datetime.strptime(date_time_str, "%m-%d-%Y %I:%M%p").replace(tzinfo=ZoneInfo("America/Chicago"))


def load_quotes_from_csv(csv_path: str, config: Config):
    state = MarketState()
    strikes = []

    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            strike_str = row.get('Strike', '')
            if not strike_str or 'Downloaded' in strike_str:
                continue

            option_type = row.get('Type')
            if option_type == 'Call':
                opt_char = 'C'
            elif option_type == 'Put':
                opt_char = 'P'
            else:
                continue

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

    return state, strikes, symbol_map


def run_snapshot(csv_path: str, yield_10y: float = 0.0, plot_path: str = None):
    config = Config()
    config.reference_time = parse_timestamp(csv_path)

    state, strikes, symbol_map = load_quotes_from_csv(csv_path, config)
    state.yield_10y = yield_10y if yield_10y > 0 else 0.0

    underlying = state.underlying_price
    underlying_yield = price_to_yield(underlying)

    result = run_model(config, strikes, symbol_map, state, y10_yield=state.yield_10y)
    if result is None:
        console.print("[red]Model failed[/red]")
        return None

    header_text = (f"[bold cyan]Snapshot: {csv_path}[/bold cyan]  |  "
                   f"[green]Underlying:[/green] {underlying:.4f} ({underlying_yield:.3f}%)  |  "
                   f"[green]/TN Adj:[/green] +1bp")
    
    renderables = build_renderables(config, result, header_text, show_stats=False)
    console.print(renderables)

    # save plots to project root 
    project_root = Path(__file__).parent
    vol_path = plot_path or str(project_root / 'vol_fit.png')
    pdf_path = str(project_root / 'pdf_fit.png')
    try:
        render_vol_fit(config, result, symbol_map, state, save_path=vol_path)
        render_pdf(config, result, symbol_map, state, save_path=pdf_path)
        console.print(f"\n[dim]Vol fit saved to {vol_path}[/dim]")
        console.print(f"[dim]PDF saved to {pdf_path}[/dim]")
    except Exception as e:
        console.print("plot error: ", e)

    return result


if __name__ == "__main__":
    default_csv = str(Path(__file__).parent / "data" / "TN_Dec10.csv")
    csv_file = sys.argv[1] if len(sys.argv) > 1 else default_csv
    yield_10y = float(sys.argv[2]) if len(sys.argv) > 2 else 4.20
    run_snapshot(csv_file, yield_10y=yield_10y)
