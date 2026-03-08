"""
Binance Futures Pump/Dump Predictor — Main Entry Point.

РЕЖИМЫ:
  scan        — Сканировать все Binance Futures на волатильные щитки
  dual-train  — Параллельное обучение PUMP (GPU0) / DUMP (GPU1)
  collect     — Только скачать данные
  live        — Live inference (требует обученных моделей)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from config import (
    PipelineConfig, GPUConfig, ModelType,
    get_rtx3060_config, get_tesla_p100_config, get_cpu_config,
)
from data_collector import (
    BinanceDataCollector,
    collect_training_data,
    load_labels,
    get_unique_symbols_from_labels,
)
from feature_engineering import (
    FeatureEngineer,
    PriceActionFeatures,
    prepare_training_data,
)
from model_training import (
    PumpDumpModel,
    train_model,
)
from dual_gpu_config import DualGPUConfig, DualGPUMetrics, print_gpu_assignment
from dual_gpu_trainer import DualGPUTrainer
from dual_gpu_inference import (
    run_dual_gpu_inference,
    example_dual_alert_handler,
)

# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: str = "logs/pipeline.log") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def get_config_for_gpu(gpu_id: int) -> PipelineConfig:
    if gpu_id == 0:
        return get_rtx3060_config()
    return get_tesla_p100_config()


def parse_symbols_arg(symbols_str: str) -> list:
    """
    Парсит аргумент --symbols в виде строки с запятыми.
    
    Args:
        symbols_str: Строка вида "Q/USDT, RIVER/USDT, DYDX/USDT"
    
    Returns:
        Список символов: ["Q/USDT", "RIVER/USDT", "DYDX/USDT"]
    """
    if not symbols_str:
        return []
    
    # Разделяем по запятым и удаляем пробелы
    symbols = [symbol.strip() for symbol in symbols_str.split(',')]
    return symbols


def load_manual_labels(
    labels_path: str,
    data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Load manual pump/dump labels from CSV and expand to per-candle labels.

    CSV format:
        symbol,start,end,type
        PEPE/USDT,2026-01-15 10:00,2026-01-15 10:30,pump
        WIF/USDT,2026-01-20 14:00,2026-01-20 14:15,dump

    All candles in [start, end] range get label 1 (pump) or -1 (dump).
    All other candles get label 0 (neutral).
    """
    events = pd.read_csv(labels_path)
    required_cols = {'symbol', 'start', 'end', 'type'}
    if not required_cols.issubset(set(events.columns)):
        missing = required_cols - set(events.columns)
        raise ValueError(
            f"Manual labels CSV missing columns: {missing}. "
            f"Required: symbol, start, end, type"
        )

    events['start'] = pd.to_datetime(events['start'])
    events['end'] = pd.to_datetime(events['end'])
    events['type'] = events['type'].str.strip().str.lower()

    all_labels = []
    symbols_with_events = set(events['symbol'].unique())

    for symbol, df in data.items():
        if symbol == "BTC/USDT":
            continue

        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        labels = pd.DataFrame({
            "timestamp": df["timestamp"],
            "symbol": symbol,
            "label": 0,
        })

        sym_events = events[events['symbol'] == symbol]
        n_pump = 0
        n_dump = 0
        for _, ev in sym_events.iterrows():
            mask = (df['timestamp'] >= ev['start']) & (df['timestamp'] <= ev['end'])
            if ev['type'] == 'pump':
                labels.loc[mask, 'label'] = 1
                n_pump += mask.sum()
            elif ev['type'] == 'dump':
                labels.loc[mask, 'label'] = -1
                n_dump += mask.sum()
            else:
                print(f"  WARNING: Unknown event type '{ev['type']}' for {symbol}, skipping")

        all_labels.append(labels)
        n_total = len(labels)
        print(
            f"  {symbol}: Pump candles={n_pump}  Dump candles={n_dump}  "
            f"Neutral={n_total - n_pump - n_dump}  Total={n_total}"
        )

    if not all_labels:
        return pd.DataFrame(columns=["timestamp", "symbol", "label"])

    result = pd.concat(all_labels, ignore_index=True)
    n_p = int((result["label"] == 1).sum())
    n_d = int((result["label"] == -1).sum())
    n_n = int((result["label"] == 0).sum())
    n = len(result)

    print(f"\nManual labels loaded: total={n}")
    print(f"  Pumps (1):   {n_p:>6} ({100*n_p/n:.1f}%)")
    print(f"  Dumps (-1):  {n_d:>6} ({100*n_d/n:.1f}%)")
    print(f"  Neutral (0): {n_n:>6} ({100*n_n/n:.1f}%)")
    print(f"  Events from CSV: {len(events)} rows, symbols: {list(symbols_with_events)}")

    return result


def generate_labels_from_data(
    data: Dict[str, pd.DataFrame],
    profiles: Optional[List] = None,
    default_threshold_pct: float = 5.0,
    lookahead: int = 5,
) -> pd.DataFrame:
    """
    Генерировать метки на основе реальных движений цены.
    Использует динамический порог (ATR) или фиксированный.
    """
    all_labels = []

    for symbol, df in data.items():
        if symbol == "BTC/USDT":  # Skip BTC
            continue

        # Динамический порог: 2.5 * ATR(14) / Close
        try:
            atr = PriceActionFeatures.atr(df, 14)
            dyn_thr = 2.5 * atr / df['close']
            threshold = dyn_thr.mean()
            if pd.isna(threshold) or threshold == 0:
                threshold = default_threshold_pct / 100.0
        except:
            threshold = default_threshold_pct / 100.0

        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        df["future_return"] = df["close"].pct_change(lookahead).shift(-lookahead)

        labels = pd.DataFrame({
            "timestamp": df["timestamp"],
            "symbol":    symbol,
            "label":     0,
        })
        
        # Применяем порог
        labels.loc[df["future_return"] > threshold,  "label"] = 1
        labels.loc[df["future_return"] < -threshold, "label"] = -1

        labels = labels.iloc[:-lookahead].dropna()
        all_labels.append(labels)

        # Логирование
        n_pumps = int((labels["label"] == 1).sum())
        n_dumps  = int((labels["label"] == -1).sum())
        n_total  = len(labels)
        if n_total > 0:
            print(
                f"  {symbol}: Pumps={n_pumps} ({100*n_pumps/n_total:.1f}%)  "
                f"Dumps={n_dumps} ({100*n_dumps/n_total:.1f}%)  "
                f"Thr=+/-{threshold*100:.1f}%"
            )

    if not all_labels:
        logger.error("No labels generated!")
        return pd.DataFrame(columns=["timestamp", "symbol", "label"])

    result = pd.concat(all_labels, ignore_index=True)
    n_p = int((result["label"] == 1).sum())
    n_d = int((result["label"] == -1).sum())
    n_n = int((result["label"] == 0).sum())
    n   = len(result)

    print(f"\nGenerated labels: total={n}")
    print(f"  Pumps (1):   {n_p:>6} ({100*n_p/n:.1f}%)")
    print(f"  Dumps (-1):  {n_d:>6} ({100*n_d/n:.1f}%)")
    print(f"  Neutral (0): {n_n:>6} ({100*n_n/n:.1f}%)")

    return result


# =============================================================================
# CLI argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binance Futures Pump/Dump Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Mode")

    # ── dual-train ────────────────────────────────────────────────────────────
    dual = subparsers.add_parser("dual-train", help="Parallel PUMP/DUMP training on dual GPU")
    dual.add_argument("--days",           type=int,   default=30)
    dual.add_argument("--pump-gpu",       type=int,   default=0)
    dual.add_argument("--dump-gpu",       type=int,   default=1)
    dual.add_argument("--pump-model",     type=str,   default="models/pump_detector.cbm")
    dual.add_argument("--dump-model",     type=str,   default="models/dump_detector.cbm")
    dual.add_argument("--pump-threshold", type=float, default=5.0,
                      help="Pump/dump threshold %% (default 5%%)")
    dual.add_argument("--no-gpu",         action="store_true", help="Use CPU")
    dual.add_argument("--sequential",     action="store_true", help="Sequential (not parallel)")
    dual.add_argument("--symbols",        type=str,
                      help="Comma-separated symbols (e.g. PEPE/USDT,WIF/USDT)")
    dual.add_argument("--lookahead", type=int, default=64,
                    help="Lookahead candles for label generation (default 64)")
    dual.add_argument("--labels", type=str, default=None,
                    help="Path to manual labels CSV (columns: symbol,start,end,type). "
                         "type = pump or dump. Overrides auto-generated labels.")

    # ── collect ───────────────────────────────────────────────────────────────
    collect = subparsers.add_parser("collect", help="Collect raw data only")
    collect.add_argument("--days",    type=int, default=30)
    collect.add_argument("--symbols", type=str, default="PEPE/USDT,WIF/USDT,FLOKI/USDT")

    # ── live ──────────────────────────────────────────────────────────────────
    live = subparsers.add_parser("live", help="Live inference (requires trained models)")
    live.add_argument("--pump-model", type=str, default="models/pump_detector.cbm")
    live.add_argument("--dump-model", type=str, default="models/dump_detector.cbm")
    live.add_argument("--symbols",    type=str, required=True)
    live.add_argument("--threshold",  type=float, default=0.85)

    return parser.parse_args()


# =============================================================================
# DUAL-TRAIN command
# =============================================================================

async def run_dual_training(args: argparse.Namespace) -> None:
    """Parallel training: PUMP on GPU0, DUMP on GPU1."""
    dual_config = DualGPUConfig(
        pump_gpu_id=args.pump_gpu,
        dump_gpu_id=args.dump_gpu,
        pump_model_path=args.pump_model,
        dump_model_path=args.dump_model,
        history_days=args.days,
    )
    print_gpu_assignment(dual_config)

    # ── Step 1: Определяем символы ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 1: Determining symbols")
    print(f"{'='*70}")

    if args.symbols:
        symbols = parse_symbols_arg(args.symbols)
        print(f"Using {len(symbols)} manually specified symbols")
    else:
        symbols = ["PEPE/USDT", "WIF/USDT", "FLOKI/USDT", "BONK/USDT", "DOGE/USDT"]
        print(f"Using default symbols: {symbols}")

    print(f"\nFinal symbols ({len(symbols)}): {symbols}")

    # ── Step 1.5: Collecting OI + Funding ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 1.5: Collecting Open Interest + Funding Rate")
    print(f"{'='*70}")

    from oi_collector import collect_oi_funding_batch
    
    # Convert "SIREN/USDT" -> "SIRENUSDT" for Binance API
    binance_symbols = [s.replace("/", "") for s in symbols]
    
    # Collect OI and Funding data
    oi_data_raw = await collect_oi_funding_batch(
        binance_symbols, 
        period="15m",  # 15-minute intervals for better resolution
        days=args.days
    )
    
    # Prepare OI data dictionary
    oi_data = {}
    for sym in symbols:
        if sym in oi_data_raw:
            oi_df = oi_data_raw[sym].get('oi', pd.DataFrame())
            if not oi_df.empty:
                oi_data[sym] = oi_df
                print(f"  {sym}: ✅ OI data ({len(oi_df)} rows)")
            else:
                oi_data[sym] = pd.DataFrame()
                print(f"  {sym}: ❌ No OI data")
        else:
            oi_data[sym] = pd.DataFrame()
            print(f"  {sym}: ❌ No OI data")
    
    print(f"OI data prepared for {len(oi_data)} symbols")

    # ── Step 2: Collect data ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 2: Collecting data from Binance")
    print(f"{'='*70}")

    config = get_config_for_gpu(0)
    config.history_days = args.days

    data = await collect_training_data(config, symbols)
    if not data:
        logger.error("Failed to collect data!")
        return

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 3: Feature Engineering")
    print(f"{'='*70}")

    engineer = FeatureEngineer(config)
    all_features_list = []
    btc_df = data.get(config.btc_symbol)

    for symbol, ohlcv_df in data.items():
        if symbol == config.btc_symbol:
            continue

        # Get OI data for this symbol if available
        symbol_oi_df = oi_data.get(symbol, pd.DataFrame())
        symbol_funding_df = pd.DataFrame()  # Funding not currently collected separately

        if not symbol_oi_df.empty:
            print(f"  {symbol}: ✅ OI data available ({len(symbol_oi_df)} rows).")
        else:
            print(f"  {symbol}: ❌ No OI data.")

        # Generate features (OI merging handled internally by FeatureEngineer)
        features_df = engineer.generate_all_features(
            ohlcv_df, 
            btc_df,
            oi_df=symbol_oi_df if not symbol_oi_df.empty else None,
            funding_df=symbol_funding_df if not symbol_funding_df.empty else None,
        )
        all_features_list.append(features_df)

    features_df = pd.concat(all_features_list, ignore_index=True)
    print(f"Features shape: {features_df.shape}")

    # ── Step 4: Generate Labels ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 4: Generating Labels")
    print(f"{'='*70}")

    if args.labels:
        print(f"Loading MANUAL labels from: {args.labels}")
        labels_df = load_manual_labels(args.labels, data)
    else:
        print("Using AUTO-generated labels (price movement threshold)")
        labels_df = generate_labels_from_data(
            data=data,
            default_threshold_pct=args.pump_threshold,
            lookahead=args.lookahead,
        )

    if labels_df.empty:
        logger.error("No labels generated!")
        return

    # ── Step 5: Prepare training data ─────────────────────────────────────────
    X, y = prepare_training_data(features_df, labels_df)

    if X.empty or len(y) == 0:
        logger.error("No training data after merge! Check timestamp alignment.")
        logger.error(f"  features_df shape: {features_df.shape}")
        logger.error(f"  labels_df shape:   {labels_df.shape}")
        return

    print(f"\nFinal training data: X={X.shape}, y={len(y)}")
    print(f"Label distribution:\n{y.value_counts()}")

    # ── Step 6: Parallel Training ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 5: Parallel Training on Dual GPU")
    print(f"{'='*70}")

    trainer = DualGPUTrainer(dual_config, use_gpu=not args.no_gpu)

    if args.sequential:
        metrics = trainer.train_sequential(X, y, X.columns.tolist())
    else:
        metrics = trainer.train_parallel(X, y, X.columns.tolist())

    # ── Results ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {metrics.total_wall_time:.2f}s")
    seq_time = metrics.pump_training_time + metrics.dump_training_time
    speedup  = seq_time / max(metrics.total_wall_time, 0.001)
    print(f"Speedup:    {speedup:.2f}x  (parallel vs sequential)")
    print(f"\nPUMP AUC:  {metrics.pump_metrics.get('auc', 0):.4f}")
    print(f"DUMP AUC:  {metrics.dump_metrics.get('auc', 0):.4f}")
    print(f"\nModels saved:")
    print(f"  PUMP: {dual_config.pump_model_path}")
    print(f"  DUMP: {dual_config.dump_model_path}")


# =============================================================================
# COLLECT command
# =============================================================================

async def run_collect(args: argparse.Namespace) -> None:
    config = get_config_for_gpu(0)
    config.history_days = args.days
    symbols = parse_symbols_arg(args.symbols)

    print(f"Collecting {args.days} days of 1m data for {len(symbols)} symbols:")
    for s in symbols:
        print(f"  {s}")

    data = await collect_training_data(config, symbols)

    print(f"\nCollected:")
    for sym, df in data.items():
        print(
            f"  {sym}: {len(df):>7} candles  "
            f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})"
        )


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    args = parse_args()
    setup_logging()

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║    Binance Futures SHITCOIN Pump/Dump Predictor  v2.2            ║
║    Dynamic Scan  |  Dual-GPU Training  |  MM Detection           ║
╚═══════════════════════════════════════════════════════════════════╝
""")

    try:
        if args.command == "dual-train":
            await run_dual_training(args)

        elif args.command == "collect":
            await run_collect(args)

        elif args.command == "live":
            symbols = [s.strip() for s in args.symbols.split(",")]
            threshold = getattr(args, 'threshold', 0.85)
            
            # Проверяем существование моделей
            import os
            pump_model = getattr(args, 'pump_model', 'models/pump_detector.cbm')
            dump_model = getattr(args, 'dump_model', 'models/dump_detector.cbm')
            
            if not os.path.exists(pump_model):
                print(f"❌ PUMP model not found: {pump_model}")
                print("   Run 'python main.py dual-train' first!")
                return
            if not os.path.exists(dump_model):
                print(f"❌ DUMP model not found: {dump_model}")
                print("   Run 'python main.py dual-train' first!")
                return
            
            print(f"✅ Models loaded successfully")
            print(f"   PUMP: {pump_model}")
            print(f"   DUMP: {dump_model}")
            print(f"   Threshold: {threshold}")
            print(f"   Symbols: {symbols}")
            
            # Запускаем live inference
            dual_config = DualGPUConfig(
                pump_model_path=pump_model,
                dump_model_path=dump_model,
                pump_alert_threshold=threshold,
                dump_alert_threshold=threshold,
            )
            
            await run_dual_gpu_inference(
                config=dual_config,
                symbols=symbols,
                alert_callback=example_dual_alert_handler,
            )
        else:
            print("\nAvailable commands:")
            print("  dual-train  — Parallel PUMP/DUMP training on dual GPU")
            print("  collect     — Download raw data only")
            print("  live        — Live inference (requires trained models)")
            print("\nExamples:")
            print("  python main.py dual-train --days 30 --symbols PEPE/USDT,WIF/USDT")
            print("  python main.py collect --days 30 --symbols PEPE/USDT,WIF/USDT")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
