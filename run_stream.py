#!/usr/bin/env python3
"""
Run realtime stream and terminal display table

Requires:
    - token.json Schwab auth token
    - .env with API_KEY and APP_SECRET
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


async def main():
    from src.stream import main as stream_main
    await stream_main()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
