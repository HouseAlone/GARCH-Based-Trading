import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from arch import arch_model

load_dotenv()
def get_alpaca_client() -> StockHistoricalDataClient:
    """
    Create a StockHistoricalDataClient using keys from .env
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if api_key is None or secret_key is None:
        raise ValueError("ALPACA_API_KEY or ALPACA_SECRET_KEY missing in .env")

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    return client


def fetch_price_data(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Alpaca for a single symbol.
    Returns a pandas DataFrame indexed by timestamp.
    """
    client = get_alpaca_client()

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=DataFeed.IEX
    )

    bars = client.get_stock_bars(request_params)
    df = bars.df

    # For single symbol, Alpaca still returns a multi-index (symbol, timestamp)
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    # Make sure it’s time-sorted
    df = df.sort_index()

    return df


def compute_returns(price_df: pd.DataFrame, column: str = "close") -> pd.Series:
    """
    Compute log-returns (or simple returns if you prefer) from price data.
    By default uses the 'close' column.
    """
    if column not in price_df.columns:
        raise ValueError(f"Column '{column}' not found in price data")

    # You can switch to simple pct_change if you prefer
    prices = price_df[column].astype(float)
    returns = (prices.pct_change().dropna())  # simple returns

    # Many people scale to % or bps; here we keep raw returns
    return returns


def fit_garch_model(returns: pd.Series):
    """
    Fit a GARCH(1,1) model to a return series using the arch package.
    Returns (model, result).
    """
    # GARCH(1,1) with constant mean and normal residuals
    model = arch_model(
        returns,
        vol="GARCH",
        p=1,
        q=1,
        mean="Constant",
        dist="normal",
    )

    result = model.fit(disp="off")  # no verbose output
    return model, result

horizon = int(os.getenv("HORIZON"))
def forecast_volatility(result, horizon) -> pd.Series:
    """
    Forecast future volatility (standard deviation) over 'horizon' steps.
    Returns a pandas Series of forecasted volatilities.
    """
    # reindex=False so it just uses integer index at the end
    forecast = result.forecast(horizon=horizon, reindex=False)

    # forecast.variance is a DataFrame: last row, all horizons
    var_series = forecast.variance.iloc[-1]
    vol_series = var_series**0.5  # convert variance -> standard deviation

    vol_series.name = "forecast_volatility"
    return vol_series


def garch_results():
    symbol = os.getenv("SYMBOL")  # change this to the Symbol in ENV

    # Example: last 3 years of daily data
    end = datetime.now()
    lookback_years = int(os.getenv("LB_YEARS"))
    lookback_months = int(os.getenv("LB_MONTHS"))
    lookback_days = int(os.getenv("LB_DAYS"))

    start = datetime(end.year - lookback_years, end.month - lookback_months, end.day - lookback_days)

    print(f"Fetching data for {symbol} from {start.date()} to {end.date()}...")
    price_df = fetch_price_data(symbol, start, end, timeframe=TimeFrame.Day)
    print(f"Got {len(price_df)} bars.")
    scale = float(os.getenv("SCALE"))
    returns = compute_returns(price_df, column="close")*scale
    print(f"Computed {len(returns)} daily returns.")

    print("\nFitting GARCH(1,1) model...")
    model, result = fit_garch_model(returns)

    print("\n=== GARCH(1,1) summary ===")
    print(result.summary())

    vol_forecast = forecast_volatility(result, horizon=horizon)
    print(f"\n{horizon}-step ahead volatility forecast (per period):")
    print(vol_forecast)


if __name__ == "__main__":
    garch_results()