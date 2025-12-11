import time
import math
import os
from datetime import timedelta
from dataclasses import dataclass

import humanize
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.columns import Columns

from .utils.math import price_to_yield, yield_to_price, decimal_to_64ths
from .utils.opt_chain import trim_strikes, fix_crossed_quotes
from .utils.pdf import build_smooth_call_prices, extract_pdf, calculate_yield_bucket_probs, get_vol, plot_vol_fit, plot_pdf
from .trading_status import MarketStatus, check_critical_status

_last_plot_time = 0
console = Console()
_live: Live | None = None


@dataclass
class ModelResult:
    rows: list
    mid: float
    T: float
    r: float
    fixed_calls: dict
    fixed_puts: dict
    vol_coeffs: tuple
    pdf: dict
    bucket_probs: dict
    stability: dict
    correction: float
    skipped: int
    interpolated: int


def init_live():
    global _live
    if _live is None:
        _live = Live(console=console, auto_refresh=False)
        _live.start()
    return _live


def stop_live():
    global _live
    if _live is not None:
        _live.stop()
        _live = None


def run_model(config, strikes, symbol_map, state, y10_yield=0.0):
    mid = state.underlying_price
    if mid == 0:
        return None
    
    rows = trim_strikes(strikes, mid, symbol_map, state)
    if len(rows) < 3:
        rows = strikes
    
    T = config.seconds_to_expiry(as_years=True)
    r = 0.045
    
    fixed_calls = fix_crossed_quotes(rows, symbol_map, state, is_call=True)
    fixed_puts = fix_crossed_quotes(rows, symbol_map, state, is_call=False)
    skipped = fixed_calls.get('_skipped', 0) + fixed_puts.get('_skipped', 0)
    interpolated = fixed_calls.get('_interpolated', 0) + fixed_puts.get('_interpolated', 0)
    
    call_prices, stability, vol_coeffs = build_smooth_call_prices(
        rows, mid, T, r, config, symbol_map, state, fixed_calls, fixed_puts
    )
    pdf = extract_pdf(call_prices, config.increment, rate=r, time_to_expiry=T)
    
    tn_yield = price_to_yield(mid)
    correction = y10_yield - tn_yield if y10_yield > 0 else 0.0
    
    bucket_probs = calculate_yield_bucket_probs(pdf, correction, config, h=config.increment)
    
    return ModelResult(
        rows=rows, mid=mid, T=T, r=r,
        fixed_calls=fixed_calls, fixed_puts=fixed_puts,
        vol_coeffs=vol_coeffs, pdf=pdf, bucket_probs=bucket_probs,
        stability=stability, correction=correction,
        skipped=skipped, interpolated=interpolated
    )


def build_renderables(config, result: ModelResult, header_text: str, status_warnings: list = None, show_stats: bool = True):
    renderables = []
    
    mid = result.mid
    rows = result.rows
    correction = result.correction

    renderables.append(Text.from_markup(f"\n{header_text}\n"))
    
    prob_sum = sum(result.pdf.get(round(k, 4), 0) for k in rows) * config.increment
    scale = 1.0 / prob_sum if prob_sum > 0 else 1.0
    
    table = Table(title="Options Chain", show_header=True, header_style="bold magenta")
    table.add_column("Strike", justify="left", style="cyan")
    table.add_column("Yield", justify="right")
    table.add_column("IV", justify="right")
    table.add_column("C.Bid", justify="right", style="green")
    table.add_column("C.Ask", justify="right", style="green")
    table.add_column("P.Bid", justify="right", style="red")
    table.add_column("P.Ask", justify="right", style="red")
    
    for K in rows:
        fixed_c = result.fixed_calls.get(K)
        fixed_p = result.fixed_puts.get(K)
        c_bid, c_ask = fixed_c if fixed_c else (0, 0)
        p_bid, p_ask = fixed_p if fixed_p else (0, 0)
        
        is_otm_call = K >= mid
        excluded = (is_otm_call and fixed_c is None) or (not is_otm_call and fixed_p is None)
        
        k = math.log(K / mid)
        iv = get_vol(k, result.vol_coeffs) if result.vol_coeffs else 0
        yld = price_to_yield(K) + correction
        
        iv_str = "[red]X[/red]" if excluded else (f"{iv*100:.1f}%" if iv > 0 else "-")
        
        table.add_row(
            f"{K:.2f}", f"{yld:.2f}%", iv_str,
            decimal_to_64ths(c_bid), decimal_to_64ths(c_ask),
            decimal_to_64ths(p_bid), decimal_to_64ths(p_ask)
        )
    
    bucket_table = Table(title="10Y Yield Probabilities", show_header=True, header_style="bold blue")
    bucket_table.add_column("Yield Interval", justify="left")
    bucket_table.add_column("TN Price Range", justify="center")
    bucket_table.add_column("Prob", justify="right", style="yellow")
    
    buckets_list = list(config.yield_buckets)
    for idx, (label, lo, hi) in enumerate(buckets_list):
        if idx == 0:
            p_high = yield_to_price(hi - correction)
            price_range = f"{p_high:.1f}-∞"
        elif idx == len(buckets_list) - 1:
            p_low = yield_to_price(lo - correction)
            price_range = f"0.0-{p_low:.1f}"
        else:
            p_high = yield_to_price(lo - correction)
            p_low = yield_to_price(hi - correction)
            price_range = f"{p_low:.1f}-{p_high:.1f}"
        prob = result.bucket_probs[label]
        bucket_table.add_row(label, price_range, f"{prob:.2f}%")
    
    renderables.append(Columns([table, bucket_table], expand=True))
    
    if show_stats:
        tn_yield = price_to_yield(mid)
        adjusted_mid = yield_to_price(tn_yield + correction) if correction else mid
        
        below = sum(result.pdf.get(round(k, 4), 0) * config.increment * scale * 100 for k in rows if k < adjusted_mid)
        above = sum(result.pdf.get(round(k, 4), 0) * config.increment * scale * 100 for k in rows if k >= adjusted_mid)
        
        fix_info = ""
        if result.interpolated > 0:
            fix_info += f"  [yellow]Interpolated: {result.interpolated}[/yellow]"
        if result.skipped > 0:
            fix_info += f"  [red]Skipped: {result.skipped}[/red]"
        
        renderables.append(Text.from_markup(
            f"  [dim]/TN adj P(< {adjusted_mid:.4f}):[/dim] [red]{below:.1f}%[/red]  |  "
            f"[dim]P(>= {adjusted_mid:.4f}):[/dim] [green]{above:.1f}%[/green]  |  "
            f"[dim]Correction:[/dim] {correction:+.2f}%  |  "
            f"[dim]IV Var:[/dim] {result.stability['iv_variance']*100:.1f}%{fix_info}"
        ))
    
    if status_warnings:
        for warning in status_warnings:
            renderables.append(Text.from_markup(warning))
    
    return Group(*renderables)


def render_vol_fit(config, result: ModelResult, symbol_map, state, save_path=None):
    plot_vol_fit(result.rows, result.mid, result.T, result.r, config, symbol_map, state, save_path=save_path)


def render_pdf(config, result: ModelResult, symbol_map, state, save_path=None):
    plot_pdf(result.rows, result.mid, result.T, result.r, config, symbol_map, state, 
             yield_correction=result.correction, save_path=save_path)


def display(config, strikes, symbol_map, state) -> MarketStatus:
    global _last_plot_time
    ts = time.strftime("%H:%M:%S")
    mid = state.underlying_price
    
    live = init_live()
    
    if mid == 0:
        live.update(Text.from_markup(f"\n  [yellow]Waiting for underlying...[/yellow] [{ts}]"), refresh=True)
        return MarketStatus(is_critical=True, reasons=["No underlying price"])
    
    result = run_model(config, strikes, symbol_map, state, y10_yield=state.yield_10y)
    if result is None:
        return MarketStatus(is_critical=True, reasons=["Model failed"])
    
    tn_yield = price_to_yield(mid)
    time_str = humanize.naturaldelta(timedelta(seconds=config.seconds_to_expiry()))
    
    header_text = (f"[bold cyan]{config.contract_root}[/bold cyan]  |  "
                   f"[green]/TN:[/green] {mid:.4f} ({tn_yield:.3f}%)  |  "
                   f"[green]/10Y:[/green] {state.yield_10y:.3f}%  |  "
                   f"Exp: {time_str}  [{ts}]")
    
    status = check_critical_status(result.stability, result.skipped, result.interpolated)
    status.bucket_probs = result.bucket_probs

    warnings = []
    if status.is_critical:
        warnings.append(f"\n  [bold red]⚠ CRITICAL: {', '.join(status.reasons)}[/bold red]")
    elif status.reasons:
        warnings.append(f"\n  [yellow]⚠ Warnings: {', '.join(status.reasons)}[/yellow]")
   
    renderables = build_renderables(config, result, header_text, warnings)
    live.update(renderables, refresh=True)
    
    if time.time() - _last_plot_time > 10:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        try:
            render_vol_fit(config, result, symbol_map, state, save_path=os.path.join(base_path, 'vol_fit.png'))
            render_pdf(config, result, symbol_map, state, save_path=os.path.join(base_path, 'pdf_fit.png'))
            _last_plot_time = time.time()
        except Exception:
            pass
    
    return status
