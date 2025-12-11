Futures Options Quotes
Level one quotes for futures options.

asyncStreamClient.level_one_futures_options_subs(symbols, *, fields=None)
Official documentation

Subscribe to level one futures options quote data.

Parameters
:
symbols – Futures options symbols to receive quotes for

fields – Iterable of LevelOneFuturesOptionsFields representing the fields to return in streaming entries. If unset, all fields will be requested.

asyncStreamClient.level_one_futures_options_unsubs(symbols)
Official documentation

Un-Subscribe to level one futures options quote data.

Parameters
:
symbols – Futures options symbols to receive quotes for

asyncStreamClient.level_one_futures_options_add(symbols, *, fields=None)
Official documentation

Add symbols to the list to receive quotes for.

Parameters
:
symbols – Futures options symbols add to list to receive quotes for

fields – Iterable of LevelOneFuturesOptionsFields representing the fields to return in streaming entries. If unset, all fields will be requested.

StreamClient.add_level_one_futures_options_handler(handler)
Register a function to handle level one futures options quotes as they are sent. See Handling Messages for details.

classStreamClient.LevelOneFuturesOptionsFields(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)
Official documentation

SYMBOL= 0
Ticker symbol in upper case.

BID_PRICE= 1
Current Bid Price

ASK_PRICE= 2
Current Ask Price

LAST_PRICE= 3
Price at which the last trade was matched

BID_SIZE= 4
Number of contracts for bid

ASK_SIZE= 5
Number of contracts for ask

BID_ID= 6
Exchange with the bid

ASK_ID= 7
Exchange with the ask

TOTAL_VOLUME= 8
Aggregated contracts traded throughout the day, including pre/post market hours.

LAST_SIZE= 9
Number of contracts traded with last trade

QUOTE_TIME_MILLIS= 10
Trade time of the last quote in milliseconds since epoch

TRADE_TIME_MILLIS= 11
Trade time of the last trade in milliseconds since epoch

HIGH_PRICE= 12
Day’s high trade price

LOW_PRICE= 13
Day’s low trade price

CLOSE_PRICE= 14
Previous day’s closing price

LAST_ID= 15
Exchange where last trade was executed

DESCRIPTION= 16
Description of the product

OPEN_PRICE= 17
Day’s Open Price

OPEN_INTEREST= 18
Open Interest

MARK= 19
Mark-to-Market value is calculated daily using current prices to determine profit/loss

TICK= 20
Minimum price movement

TICK_AMOUNT= 21
Minimum amount that the price of the market can change

FUTURE_MULTIPLIER= 22
Point value

FUTURE_SETTLEMENT_PRICE= 23
Closing price

UNDERLYING_SYMBOL= 24
Underlying symbol

STRIKE_PRICE= 25
Strike Price

FUTURE_EXPIRATION_DATE= 26
Expiration date of this contract

EXPIRATION_STYLE= 27
Expiration Style

CONTRACT_TYPE= 28
Contract Type

SECURITY_STATUS= 29
Security Status

EXCHANGE_ID= 30
Exchange character

EXCHANGE_NAME= 31
Display name of exchange