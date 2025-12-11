import math
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simpson
from .math import scholes_price, implied_vol, yield_to_price
from ..trading_status import check_iv_stability


def svi_variance(log_moneyness, a, b, rho, sigma):
    return a + b * (rho * log_moneyness + np.sqrt(log_moneyness ** 2 + sigma ** 2))


def fit_svi_params(market_data, forward):
    if len(market_data) < 3:
        return (0.04, 0.5, -0.2, 0.01)

    strikes, ivs, _ = zip(*market_data)
    log_moneyness = np.log(np.array(strikes) / forward)
    market_ivs = np.array(ivs)

    def loss(params):
        a, b, rho, sigma = params
        model_ivs = svi_variance(log_moneyness, a, b, rho, sigma)


        # helps wings to fit properly
        atm_weights = 1.0 + 5.0 * np.exp(-5.0 * log_moneyness ** 2)
        error = np.sum(atm_weights * (model_ivs - market_ivs) ** 2)

        error += 0.1 * rho ** 2  # slight skew penalty
        
        return error

    # better initial guess based on market data
    atm_iv = market_ivs[np.argmin(np.abs(log_moneyness))]
    wing_iv = np.max(market_ivs)
    initial_a = max(0.001, atm_iv - 0.02)
    initial_b = max(0.5, (wing_iv - atm_iv) * 10)  # estimate curvature from wing spread
    initial_params = [initial_a, initial_b, 0.0, 0.01]

    # wider bounds to allow proper wing fitting
    bounds = [
        (0.0, 0.15),      # a: base variance
        (0.1, 15.0),      # b: curvature 
        (-0.95, 0.95),    # rho: skew
        (0.001, 0.10)     # sigma: smoothing 
    ]

    result = minimize(loss, initial_params, method='L-BFGS-B', bounds=bounds)
    if result.success:
        return tuple(result.x)

    return (initial_a, initial_b, -0.1, 0.02)


def get_vol(log_moneyness, coeffs):
    params, _ = coeffs
    a, b, rho, sigma = params
    iv = svi_variance(log_moneyness, a, b, rho, sigma)
    return float(np.clip(iv, 0.02, 0.50))


def calculate_mid_price(bid, ask):
    if bid > 0 and ask > 0 and bid <= ask:
        return (bid ** 0.7) * (ask ** 0.3)
    elif ask > 0:
        return ask / 4
    else:
        return bid


def get_market_ivs(strikes, forward, time_to_expiry, rate, symbol_map, state, fixed_calls=None, fixed_puts=None):
    from .opt_chain import fix_crossed_quotes

    if fixed_calls is None:
        fixed_calls = fix_crossed_quotes(strikes, symbol_map, state, is_call=True)
    if fixed_puts is None:
        fixed_puts = fix_crossed_quotes(strikes, symbol_map, state, is_call=False)

    iv_data = []
    for strike in strikes:
        is_call = strike >= forward
        quotes = fixed_calls.get(strike) if is_call else fixed_puts.get(strike)

        if not quotes:
            continue

        bid, ask = quotes
        if bid <= 0 and ask <= 0:
            continue

        mid = calculate_mid_price(bid, ask)
        iv = implied_vol(mid, forward, strike, time_to_expiry, rate, is_call)

        if 0.01 < iv < 0.50:
            iv_data.append((strike, iv, 1.0))

    return iv_data


def extract_pdf(call_prices, step_size, rate=0.045, time_to_expiry=0.01):
    if time_to_expiry <= 0:
        time_to_expiry = 0.01

    discount = math.exp(rate * time_to_expiry)
    pdf = {}

    for strike in sorted(call_prices.keys()):
        strike_below = round(strike - step_size, 4)
        strike_above = round(strike + step_size, 4)

        if strike_below in call_prices and strike_above in call_prices:
            second_derivative = (call_prices[strike_below] - 2 * call_prices[strike] + call_prices[strike_above]) / (step_size ** 2)
            pdf[strike] = max(0, second_derivative * discount)

    return pdf


def build_smooth_call_prices(strikes, forward, time_to_expiry, rate, config, symbol_map, state, fixed_calls=None, fixed_puts=None):
    iv_data = get_market_ivs(strikes, forward, time_to_expiry, rate, symbol_map, state, fixed_calls, fixed_puts)
    is_stable, reason, iv_variance = check_iv_stability(iv_data, forward)

    svi_params = fit_svi_params(iv_data, forward)
    coeffs = (svi_params, time_to_expiry)

    step = config.increment
    atm_vol = get_vol(0, coeffs)
    one_stdev = atm_vol * math.sqrt(time_to_expiry) * forward

    grid_start = math.floor((forward - 5 * one_stdev) / step) * step
    grid_end = math.ceil((forward + 5 * one_stdev) / step) * step
    price_grid = np.arange(grid_start, grid_end + step, step)

    call_prices = {}
    for strike in price_grid:
        log_moneyness = math.log(strike / forward)
        vol = get_vol(log_moneyness, coeffs)
        price = scholes_price(forward, strike, time_to_expiry, rate, vol, is_call=True)
        call_prices[round(strike, 4)] = price

    stability = {
        'stable': is_stable,
        'reason': reason,
        'iv_variance': iv_variance
    }
    return call_prices, stability, coeffs


def calculate_yield_bucket_probs(pdf, yield_correction, config, h=None):
    if pdf is None:
        return {label: 0.0 for label, _, _ in config.yield_buckets}

    sorted_strikes = sorted(pdf.keys())
    strikes = np.array(sorted_strikes)
    pdf_values = np.array([pdf[k] for k in sorted_strikes])

    strike_min = strikes.min()
    strike_max = strikes.max()

    if h:
        step = h
    elif len(strikes) > 1:
        step = strikes[1] - strikes[0]
    else:
        step = 0.02

    probabilities = {}

    for label, yield_low, yield_high in config.yield_buckets:
        price_high = yield_to_price(yield_low - yield_correction) if yield_low > 0.5 else strike_max + 10
        price_low = yield_to_price(yield_high - yield_correction) if yield_high < 15 else strike_min - 10

        in_bucket = (strikes >= price_low) & (strikes <= price_high)
        num_points = in_bucket.sum()

        if num_points >= 3:
            prob = simpson(pdf_values[in_bucket], strikes[in_bucket])
        elif num_points == 2:
            prob = np.trapz(pdf_values[in_bucket], strikes[in_bucket])
        elif num_points == 1:
            prob = pdf_values[in_bucket].sum() * step
        else:
            prob = 0.0

        probabilities[label] = max(0, prob)

    total = sum(probabilities.values())
    if total > 0.01:
        # Normalize to percentages with 0.12% floor
        normalized = {label: max(0.12, (prob / total) * 100) for label, prob in probabilities.items()}
        # Renormalize to sum to 100%
        norm_total = sum(normalized.values())
        return {label: (p / norm_total) * 100 for label, p in normalized.items()}

    return {label: 100.0 / len(probabilities) for label in probabilities}


def plot_vol_fit(strikes, forward, time_to_expiry, rate, config, symbol_map, state, save_path=None):
    import matplotlib.pyplot as plt
    import time as time_module
    from .math import price_to_yield

    iv_data = get_market_ivs(strikes, forward, time_to_expiry, rate, symbol_map, state)
    if len(iv_data) < 3:
        return

    market_strikes, market_ivs, _ = zip(*iv_data)
    coeffs = (fit_svi_params(iv_data, forward), time_to_expiry)

    log_moneyness_range = np.linspace(-0.04, 0.04, 150)
    fitted_ivs = np.array([get_vol(k, coeffs) * 100 for k in log_moneyness_range])
    strike_curve = forward * np.exp(log_moneyness_range)
    atm_vol = get_vol(0, coeffs)

    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#0f0f0f')
        ax.set_facecolor('#0f0f0f')

        for i in range(len(log_moneyness_range) - 1):
            alpha = 0.08 + 0.12 * (1 - abs(log_moneyness_range[i]) / 0.04)
            ax.fill_between(strike_curve[i:i + 2], fitted_ivs[i:i + 2], alpha=alpha, color='#3b82f6')

        ax.plot(strike_curve, fitted_ivs, color='#60a5fa', lw=2)
        ax.scatter(market_strikes, np.array(market_ivs) * 100, s=50, c='#fb923c',
                   edgecolors='#0f0f0f', linewidths=0.5, zorder=5)
        ax.axvline(forward, color='#6b7280', ls='--', lw=1, alpha=0.4)
        ax.scatter([forward], [atm_vol * 100], s=40, c='#9ca3af', marker='o',
                   edgecolors='#374151', linewidths=1, alpha=0.6, zorder=6)

        tick_positions = np.arange(int(strike_curve.min()) + 1, int(strike_curve.max()), 1)
        ax.set_xticks(tick_positions)
        tick_labels = [f'{s:.0f}\n{price_to_yield(s):.2f}%' for s in tick_positions]
        ax.set_xticklabels(tick_labels)

        ax.set_xlabel('Strike / Yield', color='#e5e7eb', fontsize=10)
        ax.set_ylabel('IV %', color='#e5e7eb', fontsize=10)
        ax.tick_params(colors='#e5e7eb', labelsize=8)
        ax.spines[:].set_visible(False)
        ax.grid(alpha=0.08, color='#374151')

        title = f'{config.contract_root}  ·  Exp {config.expiry.strftime("%b %d, %Y")}  ·  ATM {atm_vol * 100:.1f}%  ·  {time_module.strftime("%m/%d %H:%M:%S")}'
        ax.set_title(title, color='#e5e7eb', fontsize=12, pad=10)

        plt.tight_layout(pad=0.5)
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f0f', bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_pdf(strikes, forward, time_to_expiry, rate, config, symbol_map, state, yield_correction=0.0, save_path=None):
    import matplotlib.pyplot as plt
    import time as time_module
    from .math import price_to_yield

    call_prices, stability, coeffs = build_smooth_call_prices(
        strikes, forward, time_to_expiry, rate, config, symbol_map, state
    )

    if not call_prices:
        return

    pdf = extract_pdf(call_prices, config.increment, rate, time_to_expiry)
    if not pdf:
        return

    sorted_strikes = sorted(pdf.keys())
    strike_arr = np.array(sorted_strikes)
    pdf_arr = np.array([pdf[k] for k in sorted_strikes])

    total_mass = simpson(pdf_arr, strike_arr) if len(pdf_arr) >= 3 else np.trapz(pdf_arr, strike_arr)
    if total_mass > 0:
        pdf_arr = pdf_arr / total_mass

    atm_vol = get_vol(0, coeffs)

    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#0f0f0f')
        ax.set_facecolor('#0f0f0f')

        for i in range(len(strike_arr) - 1):
            dist_from_fwd = abs(strike_arr[i] - forward) / forward
            alpha = 0.15 + 0.35 * np.exp(-50 * dist_from_fwd ** 2)
            ax.fill_between(strike_arr[i:i + 2], pdf_arr[i:i + 2], alpha=alpha, color='#22c55e')

        ax.plot(strike_arr, pdf_arr, color='#4ade80', lw=2)
        ax.axvline(forward, color='#6b7280', ls='--', lw=1, alpha=0.5, label='Forward')

        for label, yield_low, yield_high in config.yield_buckets:
            if 1.0 < yield_low < 10.0:
                price_boundary = yield_to_price(yield_low - yield_correction)
                if strike_arr.min() < price_boundary < strike_arr.max():
                    ax.axvline(price_boundary, color='#374151', ls=':', lw=0.5, alpha=0.4)

        tick_positions = np.arange(int(strike_arr.min()) + 1, int(strike_arr.max()), 1)
        ax.set_xticks(tick_positions)
        tick_labels = [f'{s:.0f}\n{price_to_yield(s):.2f}%' for s in tick_positions]
        ax.set_xticklabels(tick_labels)

        ax.set_xlabel('Futures Price (top) / 10Y Yield (bottom)', color='#e5e7eb', fontsize=10)
        ax.set_ylabel('Probability Density', color='#e5e7eb', fontsize=10)
        ax.tick_params(colors='#e5e7eb', labelsize=8)
        ax.spines[:].set_visible(False)
        ax.grid(alpha=0.08, color='#374151')

        ax.set_ylim(bottom=0)
        ax.set_xlim(strike_arr.min(), strike_arr.max())

        stability_txt = "✓ Stable" if stability['stable'] else f"⚠ {stability['reason']}"
        title = f'{config.contract_root}  ·  Implied PDF  ·  ATM σ {atm_vol * 100:.1f}%  ·  {time_module.strftime("%m/%d %H:%M:%S")}  ·  {stability_txt}'
        ax.set_title(title, color='#e5e7eb', fontsize=12, pad=10)

        plt.tight_layout(pad=0.5)
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f0f', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
