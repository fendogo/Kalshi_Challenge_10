# Schwab streaming for TN futures options

import asyncio, logging, os
from dotenv import load_dotenv
from schwab.auth import client_from_token_file
from schwab.streaming import StreamClient

from .config import Config, MarketState
from .utils.opt_chain import generate_strikes, generate_symbols, generate_symbol_map
from .table_display import display

load_dotenv()
logger = logging.getLogger(__name__)

TOKEN_PATH = "token.json"
API_KEY, APP_SECRET = os.getenv("API_KEY"), os.getenv("APP_SECRET")

CONFIG = Config()
STRIKES = generate_strikes(CONFIG)
SYMBOLS = generate_symbols(CONFIG, STRIKES)
SYMBOL_MAP = generate_symbol_map(CONFIG, STRIKES)
STATE = MarketState()


async def main():
    client = client_from_token_file(TOKEN_PATH, API_KEY, APP_SECRET)
    account_id = client.get_account_numbers().json()[0]['accountNumber']
    stream = StreamClient(client, account_id=account_id)
    await stream.login()
    logger.info(f"Streaming {len(SYMBOLS)} options + {CONFIG.underlying}")

    def refresh(_): display(CONFIG, STRIKES, SYMBOL_MAP, STATE)

    async def on_option(msg):
        for item in msg.get('content', []):
            if sym := item.get('key'): STATE.update_quote(sym, item)
        refresh(msg)

    async def on_underlying(msg):
        for item in msg.get('content', []):
            key = item.get('key')
            if key == CONFIG.underlying: 
                last_price = item.get('LAST_PRICE')
                if last_price and last_price > 0:
                    STATE.underlying_price = last_price
            elif key == CONFIG.yield_benchmark: STATE.update_10y(item.get('LAST_PRICE'))
        refresh(msg)

    stream.add_level_one_futures_options_handler(on_option)
    stream.add_level_one_futures_handler(on_underlying)

    F = StreamClient.LevelOneFuturesOptionsFields
    await stream.level_one_futures_options_subs(SYMBOLS, fields=[F.BID_PRICE, F.ASK_PRICE])
    await stream.level_one_futures_subs(
        [CONFIG.underlying, CONFIG.yield_benchmark],
        fields=[StreamClient.LevelOneFuturesFields.LAST_PRICE]
    )

    while True:
        try:
            await stream.handle_message()
        except Exception as e:
            logger.error(f"Disconnected: {e}")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
