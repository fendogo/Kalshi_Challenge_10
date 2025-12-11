"""
Trading status and pause criteria for market making. Goal is to prevent
pricing bugs that can lead to adverse selection or other issues. 
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class MarketStatus:
    is_critical: bool = False
    reasons: list = field(default_factory=list)
    bucket_probs: dict = None
    iv_variance: float = 0.0
    skipped: int = 0
    interpolated: int = 0


IV_VARIANCE_MAX = 0.30 # %  
MIN_IV_POINTS = 10
MAX_QUOTE_FIXES = 4
ATM_IV_MIN = 0.03 # %           
ATM_IV_MAX = 0.10           
MAX_STRIKE_GAP = 1.0


def check_iv_stability(data: list, forward_price: float) -> Tuple[bool, Optional[str], float]:
    
    if len(data) < MIN_IV_POINTS:
        error_message = f"Too few points ({len(data)})"
        return False, error_message, 0.0
    
    strike_prices = [point[0] for point in data]
    implied_volatilities = np.array([point[1] for point in data])
    
    iv_mean = np.mean(implied_volatilities)
    iv_std = np.std(implied_volatilities)
    
    has_valid_mean = iv_mean > 0
    iv_variance = iv_std / iv_mean if has_valid_mean else 0.0
    
    strike_distances_from_forward = np.abs(np.array(strike_prices) - forward_price)
    closest_strike_index = np.argmin(strike_distances_from_forward)
    atm_implied_volatility = implied_volatilities[closest_strike_index]
    
    is_atm_iv_in_range = ATM_IV_MIN < atm_implied_volatility < ATM_IV_MAX
    if not is_atm_iv_in_range:
        error_message = f"ATM IV {atm_implied_volatility*100:.1f}% out of range"
        return False, error_message, iv_variance
    
    sorted_strikes = sorted(strike_prices)
    gaps_between_strikes = np.diff(sorted_strikes)
    largest_gap = np.max(gaps_between_strikes)
    
    if largest_gap > MAX_STRIKE_GAP:
        error_message = f"Strike gap {largest_gap:.1f}"
        return False, error_message, iv_variance
    
    if iv_variance > IV_VARIANCE_MAX:
        error_message = f"High IV variance ({iv_variance*100:.0f}%)"
        return False, error_message, iv_variance
    
    return True, None, iv_variance


def check_critical_status(stability: dict, skipped: int, interpolated: int) -> MarketStatus:
    critical_reasons = []
    
    iv_variance = stability.get('iv_variance', 0.0)
    
    if iv_variance > IV_VARIANCE_MAX:
        reason = f"IV Var {iv_variance*100:.0f}% > {IV_VARIANCE_MAX*100:.0f}%"
        critical_reasons.append(reason)
    
    is_stable = stability.get('stable', True)
    stability_reason = stability.get('reason', '')
    has_too_few_points = "Too few points" in stability_reason
    
    if not is_stable and has_too_few_points:
        critical_reasons.append(stability_reason)
    
    total_fixes = skipped + interpolated
    if total_fixes > MAX_QUOTE_FIXES:
        reason = f"Fixes {total_fixes} > {MAX_QUOTE_FIXES}"
        critical_reasons.append(reason)
    
    has_critical_issues = len(critical_reasons) > 0
    
    return MarketStatus(
        is_critical=has_critical_issues,
        reasons=critical_reasons,
        iv_variance=iv_variance,
        skipped=skipped,
        interpolated=interpolated
    )


def display_status_warnings(console, stability: dict, status: MarketStatus):
    
    is_stable = stability.get('stable', True)
    
    if not is_stable:
        reason = stability.get('reason', '')
        console.print(f"  [yellow]Unstable: {reason}[/yellow]")
    
    if status.is_critical:
        console.print(f"[red]Pull orders[/red]")
