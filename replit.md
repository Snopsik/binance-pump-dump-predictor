# Binance Futures Pump/Dump Predictor v2.2

## Overview

A production-ready Python CLI pipeline for predicting sharp price movements ("pumps" and "dumps") on Binance Futures using machine learning (CatBoost / XGBoost).

## Tech Stack

- **Language**: Python 3.12
- **ML**: CatBoost, XGBoost, scikit-learn (with calibration support)
- **Data**: pandas, numpy, pyarrow, fastparquet
- **API**: aiohttp (async HTTP), websockets (real-time order book)
- **Binance API**: Authenticated via `BINANCE_API_KEY` and `BINANCE_API_SECRET` env secrets

## Project Layout

```
main.py                  # CLI entry point
config.py                # Pipeline configuration (GPU, API, hyperparams)
dual_gpu_config.py       # Dual-GPU configuration
data_collector.py        # OHLCV data collection from Binance
oi_collector.py          # Open Interest + Funding Rate collection
feature_engineering.py   # 40+ quant features (trade flow, volume anomalies, RSI, BB, VWAP, etc.)
model_training.py        # Model training with purged cross-validation
dual_gpu_trainer.py      # Parallel dual-GPU training with purged CV + feature selection
live_inference.py        # Single-model live inference
dual_gpu_inference.py    # Dual-model live inference (CPU sequential mode supported)
shitcoin_scanner.py      # Volatile coin scanner
utils.py                 # Utilities
models/                  # Saved CatBoost models (.cbm) + feature importance CSVs
data_cache/              # Parquet data cache
logs/                    # Pipeline logs and alert logs (alerts.jsonl)
```

## CLI Commands

```bash
# Download historical data only
python3 main.py collect --days 30 --symbols PEPE/USDT,WIF/USDT,FLOKI/USDT

# Train pump + dump models (parallel, CPU mode without GPU)
python3 main.py dual-train --days 30 --symbols PEPE/USDT,WIF/USDT --no-gpu

# Run live inference (requires trained models)
python3 main.py live --symbols PEPE/USDT,WIF/USDT --threshold 0.85
```

## Secrets

- `BINANCE_API_KEY` — Binance API key (stored in Replit Secrets)
- `BINANCE_API_SECRET` — Binance API secret (stored in Replit Secrets)

## ML Features (v2.2 improvements)

### Feature Groups (38+ features with _lag1 suffix)
- **Trade Flow**: taker_buy_ratio, CVD (10/30), delta_ma_ratio, aggressive_trade_size
- **Volume Anomaly**: volume z-scores (20/60/120), rvol_seasonal, vol_acceleration, vol_spikes
- **Price Action**: log_return, Parkinson vol, ATR, breakouts, wick ratios, candle body ratio
- **New in v2.2**: RSI (7/14), Bollinger Band width + %B, VWAP deviation, ROC (5/10/30), trade count z-score
- **Market Regime**: BTC correlation, correlation drop, relative strength, ATR spread
- **Open Interest**: OI change %, OI z-score, OI acceleration, OI momentum

### Training Improvements
- Purged time series cross-validation (64 candle gap to prevent look-ahead bias)
- Automatic zero-variance feature removal during training
- Optimal threshold search via precision-recall curve

### Live Inference Improvements
- Sequential CPU inference (no GPU required)
- Automatic feature alignment with trained model expectations
- Verbose error output (errors printed to console, not just logged)
- Minimum 200 candle buffer before first inference

## Workflow

The "Start application" workflow runs `python3 main.py` (shows usage). Run specific commands via the Shell for data collection, training, or live inference.
