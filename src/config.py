import logging
from datetime import date, datetime, time as dt_time
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Kalshi market 3bp yield intervals 

YIELD_BUCKETS = (
    ("3.92% or below", 0.0, 3.93),
    ("3.93% to 3.95%", 3.93, 3.96),
    ("3.96% to 3.98%", 3.96, 3.99),
    ("3.99% to 4.01%", 3.99, 4.02),
    ("4.02% to 4.04%", 4.02, 4.05),
    ("4.05% to 4.07%", 4.05, 4.08),
    ("4.08% to 4.1%", 4.08, 4.11),
    ("4.11% to 4.13%", 4.11, 4.14),
    ("4.14% to 4.16%", 4.14, 4.17),
    ("4.17% to 4.19%", 4.17, 4.20),
    ("4.2% to 4.22%", 4.20, 4.23),
    ("4.23% to 4.25%", 4.23, 4.26),
    ("4.26% to 4.28%", 4.26, 4.29),
    ("4.29% to 4.31%", 4.29, 4.32),
    ("4.32% or above", 4.32, 101.0),
)

@dataclass
class Config:
    start_strike: float = 105.0
    end_strike: float = 120.0
    increment: float = 0.25

    exchange: str = "" 
    underlying: str = "/TN"
    contract_root: str = "./TN2Z25"
    yield_benchmark: str = "/10Y"
    expiry: date = date(2025, 12, 12)
    
    yield_buckets: tuple = YIELD_BUCKETS
    reference_time: datetime = None
    
    def seconds_to_expiry(self, as_years: bool = False) -> float:
        
        et = ZoneInfo("America/New_York")
        expiry_dt = datetime.combine(self.expiry, dt_time(17, 0), tzinfo=et)
        
        if self.reference_time:
            now = self.reference_time.astimezone(et) if self.reference_time.tzinfo else self.reference_time.replace(tzinfo=et)
        else:
            now = datetime.now(et)
        
        secs = (expiry_dt - now).total_seconds()
        
        if secs <= 0:
            logger.warning(f"Options expired {secs:.0f} seconds ago")
        
        if as_years:
            seconds_per_year = 365.0 * 24 * 3600
            return max(1, secs) / seconds_per_year
        return secs


@dataclass
class MarketState:
    quotes: dict[str, dict[str, float]] = field(default_factory=dict)
    underlying_price: float = 0.0
    yield_10y: float = 0.0  

    def update_quote(self, symbol: str, data: dict) -> None:
        if symbol not in self.quotes:
            self.quotes[symbol] = {}
        for k in ('BID_PRICE', 'ASK_PRICE', 'LAST_PRICE'):
            if k in data:
                self.quotes[symbol][k] = data[k]
    
    def update_10y(self, last_price) -> None:
        if last_price and last_price > 0:
            self.yield_10y = last_price
    
    def update_underlying(self, price) -> None:
        if price:
            self.underlying_price = price

    def get_mid(self, symbol: str) -> float:
        q = self.quotes.get(symbol, {})
        bid, ask = q.get('BID_PRICE'), q.get('ASK_PRICE')
        return (bid + ask) / 2 if bid is not None and ask is not None else 0.0

    def get_bid(self, symbol: str) -> float:
        return self.quotes.get(symbol, {}).get('BID_PRICE', 0.0)

