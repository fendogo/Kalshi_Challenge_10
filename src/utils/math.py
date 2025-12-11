import math
from datetime import date

"""
CTD Bond Parameters for TNH6 (March 2026 Ultra 10Y)
Update these when the contract rolls or CTD changes
See 10Y_Ultra_Specs.pdf for contract details
See CME_CTD.png for CTD info
"""

CTD_COUPON = 4.25
CTD_MATURITY = date(2035, 8, 15)
DELIVERY_START = date(2026, 3, 1)

def get_bond_periods(settlement: date, maturity: date) -> float:
    """
    Calculate semi-annual periods for bond formula using CME rounding convention
    
    Per CME CF formula:
    - n = whole years from settlement to maturity
    - z = remaining months, rounded down to nearest quarter (0, 3, 6, 9)
    - v = z if z < 7, else 3
    
    The CME formula uses:
    - a = 1/1.03^(v/6) for fractional period
    - c = 1/1.03^(2n) for whole periods (when z < 7)
    
    Combined discounting: a × c = 1/1.03^(v/6 + 2n)
    So total periods = 2n + v/6
    
    See - CME_CF_formula.png 
    """
    total_months = (maturity.year - settlement.year) * 12 + (maturity.month - settlement.month)
    whole_years = total_months // 12
    remaining_months = total_months % 12

    z = (remaining_months // 3) * 3
    v = z if z < 7 else 3

    base_periods = 2 * whole_years if z < 7 else 2 * whole_years + 1
    return base_periods + v / 6


def calculate_conversion_factor(coupon: float, delivery_start: date, maturity: date) -> float:
    """
    Calculate CME Treasury Futures Conversion Factor using official formula.
    
    CF = a × [(coupon/2) + c + d] - b
    
    Where:
    - n = whole years from delivery month start to maturity
    - z = remaining months, rounded down to nearest quarter (0, 3, 6, 9)
    - v = z if z < 7, else 3
    - a = 1 / 1.03^(v/6)
    - b = (coupon/2) × (6 - v) / 6
    - c = 1 / 1.03^(2n) if z < 7, else 1 / 1.03^(2n+1)
    - d = (coupon/0.06) × (1 - c)
    
    See - CME_CF_formula.png
    """
    coupon_decimal = coupon / 100 
    
    total_months = (maturity.year - delivery_start.year) * 12 + (maturity.month - delivery_start.month)
    whole_years = total_months // 12
    remaining_months = total_months % 12

    z = (remaining_months // 3) * 3
    v = z if z < 7 else 3

    a = 1 / (1.03 ** (v / 6))
    b = (coupon_decimal / 2) * (6 - v) / 6
    c = 1 / (1.03 ** (2 * whole_years)) if z < 7 else 1 / (1.03 ** (2 * whole_years + 1))
    d = (coupon_decimal / 0.06) * (1 - c)

    cf = a * ((coupon_decimal / 2) + c + d) - b
    return round(cf, 4)


CTD_CF = calculate_conversion_factor(CTD_COUPON, DELIVERY_START, CTD_MATURITY)


def normal_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def scholes_price(forward, strike, time_to_expiry, rate, vol, is_call=True):
    if vol <= 0 or time_to_expiry <= 0 or forward <= 0 or strike <= 0:
        return 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * vol ** 2 * time_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    discount = math.exp(-rate * time_to_expiry)

    if is_call:
        return discount * (forward * normal_cdf(d1) - strike * normal_cdf(d2))
    else:
        return discount * (strike * normal_cdf(-d2) - forward * normal_cdf(-d1))


def implied_vol(price, forward, strike, time_to_expiry, rate, is_call=True):
    if price <= 0 or time_to_expiry <= 0 or forward <= 0 or strike <= 0:
        return 0.001

    vol = 0.1

    for _ in range(50):
        model_price = scholes_price(forward, strike, time_to_expiry, rate, vol, is_call)
        error = price - model_price

        if abs(error) < 1e-8:
            return vol

        sqrt_t = math.sqrt(time_to_expiry)
        d1 = (math.log(forward / strike) + 0.5 * vol ** 2 * time_to_expiry) / (vol * sqrt_t)
        vega = forward * math.exp(-rate * time_to_expiry) * normal_pdf(d1) * sqrt_t

        if abs(vega) < 1e-10:
            break

        vol += error / vega
        vol = max(0.001, min(5.0, vol))

    return max(0.001, vol)


def price_to_yield(futures_price: float) -> float:
    """
    Convert /TN futures price to yield using CME conversion factor formula
    
    See - CME_fut_conversion_factor.png 
    """
    if futures_price <= 0 or futures_price > 200:
        return 0.0

    bond_price = futures_price * CTD_CF
    num_periods = get_bond_periods(DELIVERY_START, CTD_MATURITY)
    coupon_payment = CTD_COUPON / 2.0

    ytm = CTD_COUPON / bond_price * 100
    ytm = max(0.1, min(20.0, ytm))

    for _ in range(30):
        semi_rate = ytm / 100.0 / 2.0

        if semi_rate > 0:
            pv_coupons = coupon_payment * (1 - (1 + semi_rate) ** (-num_periods)) / semi_rate
        else:
            pv_coupons = coupon_payment * num_periods
        pv_face = 100 * (1 + semi_rate) ** (-num_periods)
        calculated_price = pv_coupons + pv_face

        error = calculated_price - bond_price
        if abs(error) < 0.0001:
            return ytm

        delta = 0.001
        semi_rate_up = (ytm + delta) / 100.0 / 2.0
        pv_coupons_up = coupon_payment * (1 - (1 + semi_rate_up) ** (-num_periods)) / semi_rate_up if semi_rate_up > 0 else coupon_payment * num_periods
        pv_face_up = 100 * (1 + semi_rate_up) ** (-num_periods)
        price_up = pv_coupons_up + pv_face_up

        sensitivity = (price_up - calculated_price) / delta
        if abs(sensitivity) < 1e-8:
            break

        ytm -= error / sensitivity
        ytm = max(0.0, min(50.0, ytm))

    return ytm


def yield_to_price(yield_pct: float) -> float:
    if yield_pct <= 0 or yield_pct > 20:
        return 0.0

    num_periods = get_bond_periods(DELIVERY_START, CTD_MATURITY)
    coupon_payment = CTD_COUPON / 2.0
    semi_rate = yield_pct / 100.0 / 2.0

    if semi_rate > 0:
        pv_coupons = coupon_payment * (1 - (1 + semi_rate) ** (-num_periods)) / semi_rate
    else:
        pv_coupons = coupon_payment * num_periods
    pv_face = 100 * (1 + semi_rate) ** (-num_periods)
    bond_price = pv_coupons + pv_face

    return bond_price / CTD_CF


def decimal_to_64ths(price: float) -> str:
    if price <= 0:
        return "0\"00"

    whole = int(price)
    fractional = price - whole
    sixty_fourths = fractional * 64

    if abs(sixty_fourths - round(sixty_fourths)) < 0.01:
        ticks = round(sixty_fourths)
        if ticks >= 64:
            whole += 1
            ticks = 0
        return f"{whole}\"{ticks:02d}"
    else:
        ticks = round(sixty_fourths, 1)
        if ticks >= 64:
            whole += 1
            ticks = 0
        return f"{whole}\"{ticks:04.1f}"
