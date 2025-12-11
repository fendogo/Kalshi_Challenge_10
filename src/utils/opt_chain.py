import numpy as np
from ..config import Config


def generate_strikes(config: Config) -> list[float]:
    raw = np.arange(config.start_strike, config.end_strike + 0.01, config.increment)
    return [round(float(s), 4) for s in raw]


def generate_symbols(config: Config, strikes: list[float]) -> list[str]:
    symbols = []
    for strike in strikes:
        symbols.append(f"{config.contract_root}C{strike:g}{config.exchange}")
        symbols.append(f"{config.contract_root}P{strike:g}{config.exchange}")
    return symbols


def generate_symbol_map(config: Config, strikes: list[float]) -> dict:
    symbol_map = {}
    for strike in strikes:
        call = f"{config.contract_root}C{strike:g}{config.exchange}"
        put = f"{config.contract_root}P{strike:g}{config.exchange}"
        symbol_map[strike] = (call, put)
    return symbol_map


def find_nearest_index(values: list[float], target: float) -> int:
    differences = np.abs(np.array(values) - target)
    return int(np.argmin(differences))


def calculate_underlying_from_parity(strikes, symbol_map, state, yield_adjustment_bp=0.0):
    estimates = []
    for strike in strikes:
        call_sym, put_sym = symbol_map[strike]
        call_quote = state.quotes.get(call_sym, {})
        put_quote = state.quotes.get(put_sym, {})

        call_mid = ((call_quote.get('BID_PRICE') or 0) + (call_quote.get('ASK_PRICE') or 0)) / 2
        put_mid = ((put_quote.get('BID_PRICE') or 0) + (put_quote.get('ASK_PRICE') or 0)) / 2

        if call_mid > 0 and put_mid > 0:
            implied_forward = call_mid - put_mid + strike
            estimates.append(implied_forward)

    if not estimates:
        return 0.0
    return sum(estimates) / len(estimates) + yield_adjustment_bp * 0.01


def trim_strikes(strikes: list[float], underlying: float, symbol_map: dict, state) -> list[float]:
    # Cropping range to strikes with bids
    nearest = find_nearest_index(strikes, underlying)
    first_valid = 0
    last_valid = len(strikes)

    for i in range(nearest, -1, -1):
        _, put_sym = symbol_map[strikes[i]]
        if state.get_bid(put_sym) == 0:
            has_bid_below = i > 0 and state.get_bid(symbol_map[strikes[i - 1]][1]) > 0
            if not has_bid_below:
                first_valid = i - 2
                break

    for i in range(nearest, len(strikes)):
        call_sym, _ = symbol_map[strikes[i]]
        if state.get_bid(call_sym) == 0:
            has_bid_above = i < len(strikes) - 1 and state.get_bid(symbol_map[strikes[i + 1]][0]) > 0
            if not has_bid_above:
                last_valid = i + 3
                break

    return strikes[first_valid:last_valid]


def fix_crossed_quotes(strikes: list[float], symbol_map: dict, state, is_call: bool = True):
    # Fix crossed quotes and zero bids using neighbor interpolation
    
    fixed = {}
    skipped = 0
    interpolated = 0

    def get_quote(strike_idx):
        sym = symbol_map[strikes[strike_idx]][0 if is_call else 1]
        return state.quotes.get(sym, {}).get('BID_PRICE', 0) or 0

    def get_neighbor_bids(idx):
        bid_below = get_quote(idx - 1) if idx > 0 else 0.0
        bid_above = get_quote(idx + 1) if idx < len(strikes) - 1 else 0.0
        return bid_below, bid_above

    def price_fits_monotonicity(price, bid_below, bid_above):
        if price <= 0:
            return False
        if is_call:
            ok_below = bid_below <= 0 or price <= bid_below * 1.1
            ok_above = bid_above <= 0 or price >= bid_above * 0.9
        else:
            ok_below = bid_below <= 0 or price >= bid_below * 0.9
            ok_above = bid_above <= 0 or price <= bid_above * 1.1
        return ok_below and ok_above

    for i, strike in enumerate(strikes):
        sym = symbol_map[strike][0 if is_call else 1]
        quote = state.quotes.get(sym, {})
        bid = quote.get('BID_PRICE', 0) or 0
        ask = quote.get('ASK_PRICE', 0) or 0

        bid_below, bid_above = get_neighbor_bids(i)
        
        # zero bid with both neighbors - interpolate
        if bid <= 0 and bid_below > 0 and bid_above > 0:
            interpolated_bid = (bid_below + bid_above) / 2
            if ask > 0 and interpolated_bid > ask:
                fixed[strike] = (ask, interpolated_bid)
            else:
                fixed[strike] = (interpolated_bid, ask if ask > 0 else interpolated_bid)
            interpolated += 1
            continue
        
        # normal quote not crossed
        if bid <= ask:
            fixed[strike] = (bid, ask)
            continue

        bid_ok = price_fits_monotonicity(bid, bid_below, bid_above)
        ask_ok = price_fits_monotonicity(ask, bid_below, bid_above)

        if bid_ok and ask_ok:
            if bid_below > 0 and bid_above > 0:
                neighbor_mid = (bid_below + bid_above) / 2
                use_bid = abs(bid - neighbor_mid) <= abs(ask - neighbor_mid)
                fixed[strike] = (bid, bid) if use_bid else (ask, ask)
            else:
                fixed[strike] = (bid, bid)
        elif bid_ok:
            fixed[strike] = (bid, bid)
        elif ask_ok:
            fixed[strike] = (ask, ask)
        else:
            fixed[strike] = None
            skipped += 1

    fixed['_skipped'] = skipped
    fixed['_interpolated'] = interpolated
    return fixed
