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

# small env helpers
def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default


# default daily timeframe
DEFAULT_TIMEFRAME: Final[TimeFrame] = TimeFrame(1, TimeFrameUnit.Day)

# universe from env (comma-separated) with a default fallback
UNIVERSE = [s.strip().upper() for s in os.getenv("UNIVERSE", "AAPL,MSFT,AMZN,GOOGL,META,NVDA,TSLA,JPM,V,HD,XOM,SPY,QQQ").split(",") if s.strip()]

# shared tunables
GARCH_P = env_int("GARCH_P", 1)
GARCH_Q = env_int("GARCH_Q", 1)
VOL_LOW_REL = env_float("VOL_LOW_REL", 0.8)
VOL_HIGH_REL = env_float("VOL_HIGH_REL", 1.2)
MR_LOOKBACK = env_int("MR_LOOKBACK", 20)
MR_Z = env_float("MR_Z", -1.0)
MOM_FAST = env_int("MOM_FAST", 12)
MOM_SLOW = env_int("MOM_SLOW", 26)
MOM_SIGNAL = env_int("MOM_SIGNAL", 9)
TREND_SMA = env_int("TREND_SMA", 50)
WEIGHT_LOW = env_float("WEIGHT_LOW", 0.02)
WEIGHT_MED = env_float("WEIGHT_MED", 0.01)


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


def load_large_cap_universe() -> list:
    # unused now that universe is fixed
    return UNIVERSE


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
        p=GARCH_P,
        q=GARCH_Q,
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
    # label vol regime using relative vol vs median
    cond_vol = pd.Series(result.conditional_volatility)
    median_vol = cond_vol.median()
    forecast = result.forecast(horizon=horizon, reindex=False)
    var_series = forecast.variance.iloc[-1]
    forecast_vol = float(var_series.iloc[-1] ** 0.5)

    v_rel = forecast_vol / median_vol if median_vol > 0 else 1.0
    if v_rel <= VOL_LOW_REL:
        regime = "low"
    elif v_rel >= VOL_HIGH_REL:
        regime = "high"
    else:
        regime = "medium"
    return regime, forecast_vol


def mean_reversion_weight(price_df: pd.DataFrame, max_weight: float = 0.02) -> float:
    # mean reversion using 20d z-score
    close = price_df["close"].astype(float)
    sma = close.rolling(MR_LOOKBACK).mean()
    std = close.rolling(MR_LOOKBACK).std()
    if pd.isna(sma.iloc[-1]) or pd.isna(std.iloc[-1]) or std.iloc[-1] == 0:
        return 0.0
    z = (close.iloc[-1] - sma.iloc[-1]) / std.iloc[-1]
    return max_weight if z < MR_Z else 0.0


def momentum_weight(price_df: pd.DataFrame, max_weight: float = 0.01) -> float:
    # trend follow with MACD plus SMA50 filter
    close = price_df["close"].astype(float)
    macd = close.ewm(span=MOM_FAST).mean() - close.ewm(span=MOM_SLOW).mean()
    signal = macd.ewm(span=MOM_SIGNAL).mean()
    sma = close.rolling(TREND_SMA).mean()
    if pd.isna(macd.iloc[-1]) or pd.isna(signal.iloc[-1]) or pd.isna(sma.iloc[-1]):
        return 0.0
    cond = macd.iloc[-1] > signal.iloc[-1] and close.iloc[-1] > sma.iloc[-1]
    return max_weight if cond else 0.0


def target_weight_from_regime(regime: str, price_df: pd.DataFrame) -> float:
    # tie regime to sizing style
    if regime == "high":  # risk-off
        return 0.0
    if regime == "medium":  # medium vol -> momentum
        return momentum_weight(price_df, max_weight=WEIGHT_MED)
    # low vol -> mean reversion
    return mean_reversion_weight(price_df, max_weight=WEIGHT_LOW)


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
            weight = target_weight_from_regime(regime, price_df)
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
