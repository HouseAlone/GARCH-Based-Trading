import os
from datetime import datetime
from typing import Dict, Tuple, Final

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from arch import arch_model

load_dotenv()  # load env vars

# simple universe to run
UNIVERSE = ["AAPL", "MSFT", "SPY"]

# default daily timeframe
DEFAULT_TIMEFRAME: Final[TimeFrame] = TimeFrame(1, TimeFrameUnit.Day)


def get_alpaca_client() -> StockHistoricalDataClient:
    # data client for historical bars
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if api_key is None or secret_key is None:
        raise ValueError("ALPACA_API_KEY or ALPACA_SECRET_KEY missing in .env")
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def get_trading_client() -> TradingClient:
    # trading client for paper trading
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if api_key is None or secret_key is None:
        raise ValueError("ALPACA_API_KEY or ALPACA_SECRET_KEY missing in .env")
    return TradingClient(api_key, secret_key, paper=True)


def fetch_price_data(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    # fetch OHLCV for one symbol
    client = get_alpaca_client()
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(request_params)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    return df.sort_index()


def compute_returns(price_df: pd.DataFrame, column: str = "close") -> pd.Series:
    # simple pct change returns
    if column not in price_df.columns:
        raise ValueError(f"Column '{column}' not found in price data")
    prices = price_df[column].astype(float)
    returns = prices.pct_change().dropna()
    return returns


def fit_garch_model(returns: pd.Series):
    # fit GARCH(1,1) with t dist
    model = arch_model(
        returns,
        vol="GARCH",
        p=1,
        q=1,
        mean="Constant",
        dist="t",
    )
    result = model.fit(disp="off")
    return model, result


def forecast_volatility(result, horizon: int) -> pd.Series:
    # forecast vol for given horizon
    forecast = result.forecast(horizon=horizon, reindex=False)
    var_series = forecast.variance.iloc[-1]
    vol_series = var_series ** 0.5
    vol_series.name = "forecast_volatility"
    return vol_series


def classify_vol_regime(result, horizon: int = 1) -> Tuple[str, float]:
    # label vol regime using percentiles
    cond_vol = pd.Series(result.conditional_volatility)
    low_p = cond_vol.quantile(0.2)
    high_p = cond_vol.quantile(0.8)
    forecast = result.forecast(horizon=horizon, reindex=False)
    var_series = forecast.variance.iloc[-1]
    forecast_vol = float(var_series.iloc[-1] ** 0.5)

    if forecast_vol < low_p:
        regime = "low"
    elif forecast_vol > high_p:
        regime = "high"
    else:
        regime = "normal"
    return regime, forecast_vol


def target_weight_from_regime(regime: str) -> float:
    # map regime to portfolio weight
    if regime == "low":
        return 0.02
    if regime == "normal":
        return 0.01
    return 0.0


def build_vol_signals_for_universe(symbols) -> Dict[str, float]:
    # build target weights for each symbol
    end = datetime.now()
    lookback_years = int(os.getenv("LB_YEARS"))
    lookback_months = int(os.getenv("LB_MONTHS"))
    lookback_days = int(os.getenv("LB_DAYS"))
    scale = float(os.getenv("SCALE"))

    start = (
        pd.Timestamp(end)
        - pd.DateOffset(years=lookback_years, months=lookback_months, days=lookback_days)
    ).to_pydatetime()

    signals: Dict[str, float] = {}
    for symbol in symbols:
        try:
            price_df = fetch_price_data(symbol, start, end, timeframe=DEFAULT_TIMEFRAME)
            returns = compute_returns(price_df, column="close") * scale
            _, result = fit_garch_model(returns)
            regime, forecast_vol = classify_vol_regime(result, horizon=1)
            weight = target_weight_from_regime(regime)
            signals[symbol] = weight
            print(f"{symbol}: regime={regime}, fvol={forecast_vol:.6f}, weight={weight:.4f}")
        except Exception as exc:
            print(f"{symbol}: error {exc}")
    return signals


def get_equity_and_positions(client: TradingClient) -> Tuple[float, Dict[str, float]]:
    # fetch account equity and current positions
    account = client.get_account()
    equity = float(account.equity)
    positions = client.get_all_positions()
    pos_map: Dict[str, float] = {}
    for pos in positions:
        pos_map[pos.symbol] = float(pos.qty)
    return equity, pos_map


def execute_signals(signals: Dict[str, float]) -> None:
    # submit market orders to move toward target weights
    client = get_trading_client()
    equity, current_positions = get_equity_and_positions(client)

    for symbol, target_weight in signals.items():
        target_dollars = equity * target_weight

        # get recent price to size orders
        end = datetime.now()
        start = (pd.Timestamp(end) - pd.DateOffset(days=14)).to_pydatetime()
        price_df = fetch_price_data(symbol, start, end, timeframe=DEFAULT_TIMEFRAME)
        if price_df.empty:
            print(f"{symbol}: no recent data, skipping")
            continue
        last_price = float(price_df["close"].iloc[-1])

        target_qty = int(target_dollars // last_price)
        current_qty = int(current_positions.get(symbol, 0))
        diff = target_qty - current_qty

        if diff == 0:
            print(f"{symbol}: already at target ({target_qty} shares)")
            continue

        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        qty = abs(diff)

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        client.submit_order(order)
        print(f"{symbol}: sent {side.value} for {qty} shares (target {target_qty}, current {current_qty})")


def garch_results():
    # debug helper for single symbol from env
    symbol = os.getenv("SYMBOL")
    end = datetime.now()
    lookback_years = int(os.getenv("LB_YEARS"))
    lookback_months = int(os.getenv("LB_MONTHS"))
    lookback_days = int(os.getenv("LB_DAYS"))
    scale = float(os.getenv("SCALE"))
    horizon = int(os.getenv("HORIZON"))

    start = (
        pd.Timestamp(end)
        - pd.DateOffset(years=lookback_years, months=lookback_months, days=lookback_days)
    ).to_pydatetime()

    print(f"Fetching data for {symbol} from {start.date()} to {end.date()}...")
    price_df = fetch_price_data(symbol, start, end, timeframe=DEFAULT_TIMEFRAME)
    print(f"Got {len(price_df)} bars.")
    returns = compute_returns(price_df, column="close") * scale
    print(f"Computed {len(returns)} daily returns.")

    print("\nFitting GARCH(1,1) model...")
    model, result = fit_garch_model(returns)

    print("\n=== GARCH(1,1) summary ===")
    print(result.summary())

    vol_forecast = forecast_volatility(result, horizon=horizon)
    print(f"\n{horizon}-step ahead volatility forecast (per period):")
    print(vol_forecast)


if __name__ == "__main__":
    signals = build_vol_signals_for_universe(UNIVERSE)
    execute_signals(signals)
