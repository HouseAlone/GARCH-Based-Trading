# GARCH Volatility Module

This project computes volatility forecasts using a **GARCH(1,1)** model with historical price data retrieved from **Alpaca**.  
It is designed as a modular volatility engine for a larger trading bot project.

---

## 📌 What the Script Does

- Loads configuration values from a `.env` file  
- Connects to Alpaca using your API keys  
- Fetches historical OHLCV data (using the free IEX feed)  
- Computes daily returns (scaled using the variable `SCALE`)  
- Fits a **GARCH(1,1)** volatility model using the `arch` package  
- Outputs multi-step volatility forecasts  
  - `h.1` = volatility 1 day ahead  
  - `h.2` = volatility 2 days ahead  
  - etc.

Because the script uses **daily** price data, each forecast step corresponds to **1 trading day**.

---

## 📦 Installation

Install all required packages:

```bash
pip install -r requirements.txt
