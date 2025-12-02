import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from arch import arch_model

# reuse data helpers from the live script
from GARCH_Model import fetch_price_data, compute_returns, DEFAULT_TIMEFRAME, UNIVERSE

load_dotenv()

# helper to read env with defaults
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


# tunable params from .env (with defaults)
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
SCALE = env_float("SCALE", 1.0)


def fit_garch_custom(returns: pd.Series):
    # fit GARCH with env-configurable order
    model = arch_model(
        returns,
        vol="GARCH",
        p=GARCH_P,
        q=GARCH_Q,
        mean="Constant",
        dist="t",
    )
    return model.fit(disp="off")


def classify_vol_regime_env(result, horizon: int = 1):
    # classify with relative vol thresholds from env
    cond_vol = pd.Series(result.conditional_volatility)
    median_vol = cond_vol.median()
    forecast = result.forecast(horizon=horizon, reindex=False)
    var_series = forecast.variance.iloc[-1]
    forecast_vol = float(var_series.iloc[-1] ** 0.5)
    v_rel = forecast_vol / median_vol if median_vol > 0 else 1.0
    if v_rel <= VOL_LOW_REL:
        return "low", forecast_vol
    if v_rel >= VOL_HIGH_REL:
        return "high", forecast_vol
    return "medium", forecast_vol


def mean_reversion_weight_env(close: pd.Series) -> float:
    # mean reversion with z-score
    if len(close) < MR_LOOKBACK:
        return 0.0
    slice_close = close.iloc[-MR_LOOKBACK:]
    sma = slice_close.mean()
    std = slice_close.std()
    if std == 0 or pd.isna(std):
        return 0.0
    z = (slice_close.iloc[-1] - sma) / std
    return WEIGHT_LOW if z < MR_Z else 0.0


def momentum_weight_env(close: pd.Series) -> float:
    # momentum with MACD and trend filter
    if len(close) < max(TREND_SMA, MOM_SLOW + MOM_SIGNAL):
        return 0.0
    macd = close.ewm(span=MOM_FAST).mean() - close.ewm(span=MOM_SLOW).mean()
    signal = macd.ewm(span=MOM_SIGNAL).mean()
    sma_trend = close.rolling(TREND_SMA).mean()
    if pd.isna(macd.iloc[-1]) or pd.isna(signal.iloc[-1]) or pd.isna(sma_trend.iloc[-1]):
        return 0.0
    cond = macd.iloc[-1] > signal.iloc[-1] and close.iloc[-1] > sma_trend.iloc[-1]
    return WEIGHT_MED if cond else 0.0


def weight_from_regime_env(regime: str, close: pd.Series) -> float:
    # map regime to sizing rule
    if regime == "high":
        return 0.0
    if regime == "medium":
        return momentum_weight_env(close)
    return mean_reversion_weight_env(close)


def backtest_symbol(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    # simple walk-forward backtest per symbol
    buffer_days = max(TREND_SMA, MR_LOOKBACK, MOM_SLOW + MOM_SIGNAL) + 10
    start_buffer = (pd.Timestamp(start) - pd.DateOffset(days=buffer_days)).to_pydatetime()
    price_df = fetch_price_data(symbol, start_buffer, end, timeframe=DEFAULT_TIMEFRAME)
    if price_df.empty:
        return pd.DataFrame()
    # drop timezone to align with naive start/end
    if hasattr(price_df.index, "tz") and price_df.index.tz is not None:
        price_df.index = price_df.index.tz_convert(None)
    price_df = price_df.loc[pd.Timestamp(start): pd.Timestamp(end)]
    close = price_df["close"].astype(float)
    rets = compute_returns(price_df, column="close") * SCALE

    rows: List[Dict] = []
    for i in range(buffer_days, len(close) - 1):
        # data up to day i
        close_slice = close.iloc[: i + 1]
        ret_slice = rets.iloc[: i]  # up to current day
        if len(ret_slice) < max(GARCH_P, GARCH_Q) + 10:
            continue
        try:
            result = fit_garch_custom(ret_slice)
        except Exception:
            continue
        regime, fvol = classify_vol_regime_env(result, horizon=1)
        weight = weight_from_regime_env(regime, close_slice)
        next_ret = rets.iloc[i] if i < len(rets) else 0.0
        pnl = weight * next_ret
        rows.append(
            {
                "date": close.index[i],
                "regime": regime,
                "weight": weight,
                "forecast_vol": fvol,
                "next_ret": next_ret,
                "pnl": pnl,
            }
        )
    return pd.DataFrame(rows)


def run_backtest(symbols: List[str], start: datetime, end: datetime):
    # run backtest across symbols
    all_results = []
    for symbol in symbols:
        print(f"Backtesting {symbol}...")
        df = backtest_symbol(symbol, start, end)
        if df.empty:
            print(f"{symbol}: no data or no trades")
            continue
        df["symbol"] = symbol
        all_results.append(df)
        total_ret = (1 + df["pnl"]).prod() - 1
        print(f"{symbol}: trades={len(df)}, total_return={total_ret:.4f}")
    if not all_results:
        print("No results to show.")
        return
    merged = pd.concat(all_results, ignore_index=True)
    portfolio = merged.groupby("date")["pnl"].mean()
    equity_curve = (1 + portfolio).cumprod()
    print(f"Portfolio final return: {equity_curve.iloc[-1] - 1:.4f}")
    return merged, equity_curve


def parse_dates():
    # parse custom backtest dates from env
    start_str = os.getenv("BACKTEST_START")
    end_str = os.getenv("BACKTEST_END")
    if not start_str or not end_str:
        raise ValueError("Set BACKTEST_START and BACKTEST_END in .env (YYYY-MM-DD)")
    return pd.Timestamp(start_str), pd.Timestamp(end_str)


def parse_symbols():
    # parse symbols from env or use shared universe
    symbols_str = os.getenv("UNIVERSE", "")
    if symbols_str.strip():
        return [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
    return UNIVERSE


if __name__ == "__main__":
    start_dt, end_dt = parse_dates()
    symbols = parse_symbols()
    run_backtest(symbols, start_dt, end_dt)
