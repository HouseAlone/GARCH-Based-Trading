from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import os
client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
req = StockBarsRequest(symbol_or_symbols="AAPL", timeframe=TimeFrame(1, TimeFrameUnit.Day), start="2024-11-15", end="2024-11-30")
print(client.get_stock_bars(req).df.head())
