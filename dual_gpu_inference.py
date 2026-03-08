"""
Параллельный Live Inference на двух GPU с WebSocket API.
Supports CPU-only mode (sequential inference) when GPU is unavailable.
"""

import asyncio
import json
import os
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import logging

import numpy as np
import pandas as pd
import aiohttp

from dual_gpu_config import DualGPUConfig, ModelTarget
from data_collector import BinanceDataCollector
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

WS_STREAM_URL = "wss://fstream.binance.com/stream"

MIN_HISTORY_CANDLES = 200


@dataclass
class DualAlert:
    timestamp: datetime
    symbol: str
    pump_probability: float
    dump_probability: float
    dominant_signal: str
    confidence: float
    price: float
    volume: float = 0.0
    features_snapshot: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'pump_probability': self.pump_probability,
            'dump_probability': self.dump_probability,
            'dominant_signal': self.dominant_signal,
            'confidence': self.confidence,
            'price': self.price,
            'volume': self.volume,
        }


def _load_model_feature_names(model_path: str) -> Optional[List[str]]:
    fi_path = model_path.replace('.cbm', '_feature_importance.csv')
    try:
        fi_df = pd.read_csv(fi_path)
        return fi_df['feature'].tolist()
    except Exception:
        return None


class CachedModelInference:
    _pump_model = None
    _dump_model = None
    _pump_path = None
    _dump_path = None

    @classmethod
    def load(cls, pump_model_path: str, dump_model_path: str):
        from catboost import CatBoostClassifier
        if cls._pump_path != pump_model_path or cls._pump_model is None:
            cls._pump_model = CatBoostClassifier()
            cls._pump_model.load_model(pump_model_path)
            cls._pump_path = pump_model_path
            print(f"   Loaded PUMP model from {pump_model_path}", flush=True)
        if cls._dump_path != dump_model_path or cls._dump_model is None:
            cls._dump_model = CatBoostClassifier()
            cls._dump_model.load_model(dump_model_path)
            cls._dump_path = dump_model_path
            print(f"   Loaded DUMP model from {dump_model_path}", flush=True)

    @classmethod
    def predict(cls, X_dict: Dict) -> Tuple[float, float]:
        X = pd.DataFrame([X_dict['feature_values']], columns=X_dict['feature_names'])
        pump_prob = float(cls._pump_model.predict_proba(X)[0, 1])
        dump_prob = float(cls._dump_model.predict_proba(X)[0, 1])
        return pump_prob, dump_prob


def inference_sequential(pump_model_path: str, dump_model_path: str, X_dict: Dict) -> Tuple[float, float]:
    try:
        CachedModelInference.load(pump_model_path, dump_model_path)
        return CachedModelInference.predict(X_dict)
    except Exception as e:
        print(f"   INFERENCE ERROR: {e}", flush=True)
        logger.error(f"Sequential inference error: {e}", exc_info=True)
        return 0.0, 0.0


class BinanceWebSocketManager:
    def __init__(self, symbols: List[str], interval: str = "1m"):
        self.symbols = [s.upper().replace("/", "") for s in symbols]
        self.interval = interval
        self._ws = None
        self._session = None
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {}

    def on_kline(self, symbol: str, callback: Callable) -> None:
        symbol = symbol.upper().replace("/", "")
        if symbol not in self._callbacks:
            self._callbacks[symbol] = []
        self._callbacks[symbol].append(callback)

    def _get_stream_url(self) -> str:
        streams = [f"{s.lower()}@kline_{self.interval}" for s in self.symbols]
        return f"{WS_STREAM_URL}?streams={'/'.join(streams)}"

    async def connect(self) -> None:
        url = self._get_stream_url()
        logger.info(f"Connecting to WebSocket...")
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(url, heartbeat=30)
        self._running = True
        logger.info(f"WebSocket connected! Streaming {len(self.symbols)} symbols")

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    async def listen(self) -> None:
        while self._running:
            try:
                msg = await self._ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_message(data)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    async def _handle_message(self, data: Dict) -> None:
        if 'stream' not in data or 'data' not in data:
            return
        kline = data['data'].get('k', {})
        if not kline:
            return
        symbol = kline.get('s', '')
        is_closed = kline.get('x', False)
        if is_closed and symbol in self._callbacks:
            kline_info = {
                'symbol': symbol,
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']), 'high': float(kline['h']),
                'low': float(kline['l']), 'close': float(kline['c']),
                'volume': float(kline['v']), 'quote_volume': float(kline['q']),
                'num_trades': int(kline['n']),
                'taker_buy_base_volume': float(kline['V']),
                'taker_buy_quote_volume': float(kline['Q']),
            }
            for callback in self._callbacks[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(kline_info)
                    else:
                        callback(kline_info)
                except Exception as e:
                    logger.error(f"Callback error: {e}")


class DualGPUInferenceEngine:
    def __init__(self, config: DualGPUConfig, symbols: List[str], alert_callback=None):
        self.config = config
        self.symbols = symbols
        self.alert_callback = alert_callback
        self._pump_model_path = config.pump_model_path
        self._dump_model_path = config.dump_model_path
        self._data_buffer: Dict[str, deque] = {}
        self._running = False
        self._ws_manager = None
        from config import PipelineConfig
        self._pipeline_config = PipelineConfig()
        self._feature_engineer = FeatureEngineer(self._pipeline_config)
        self._stats = {'klines': 0, 'alerts': 0, 'times': []}

        self._pump_expected_features = _load_model_feature_names(self._pump_model_path)
        self._dump_expected_features = _load_model_feature_names(self._dump_model_path)

        if self._pump_expected_features:
            self._expected_features = self._pump_expected_features
            print(f"   Loaded {len(self._expected_features)} expected features from pump model", flush=True)
        elif self._dump_expected_features:
            self._expected_features = self._dump_expected_features
            print(f"   Loaded {len(self._expected_features)} expected features from dump model", flush=True)
        else:
            self._expected_features = None
            print(f"   WARNING: Could not load expected features from model metadata", flush=True)

        print(f"   Pre-loading CatBoost models...", flush=True)
        try:
            CachedModelInference.load(self._pump_model_path, self._dump_model_path)
        except Exception as e:
            print(f"   WARNING: Failed to pre-load models: {e}", flush=True)

    async def start(self) -> None:
        self._running = True
        print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                   LIVE INFERENCE ENGINE                               ║
║                    (WebSocket Real-Time)                              ║
╠═══════════════════════════════════════════════════════════════════════╣
║   PUMP Model  ───>  PUMP Probability                                  ║
║   DUMP Model  ───>  DUMP Probability                                  ║
╚═══════════════════════════════════════════════════════════════════════╝
        """, flush=True)
        print(f"   PUMP threshold: {self.config.pump_alert_threshold:.0%}", flush=True)
        print(f"   DUMP threshold: {self.config.dump_alert_threshold:.0%}", flush=True)
        print(f"   Mode: Sequential CPU inference", flush=True)
        await self._load_initial_history()
        self._ws_manager = BinanceWebSocketManager(self.symbols, interval="1m")
        for symbol in self.symbols:
            self._ws_manager.on_kline(symbol.replace("/", ""), self._on_kline_closed)
        await self._ws_manager.connect()
        print(f"\n   WebSocket listener started. Waiting for candles...", flush=True)
        try:
            await self._ws_manager.listen()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        if self._ws_manager:
            await self._ws_manager.disconnect()
        print(f"\n   Stats: {self._stats['klines']} klines, {self._stats['alerts']} alerts", flush=True)

    async def _load_initial_history(self) -> None:
        print(f"\n   Loading initial history...", flush=True)
        all_symbols = self.symbols + [self._pipeline_config.btc_symbol]
        async with BinanceDataCollector(self._pipeline_config) as collector:
            for symbol in all_symbols:
                try:
                    df = await collector.fetch_ohlcv(symbol, limit=500)
                    if df.empty:
                        print(f"   WARNING: No data for {symbol}", flush=True)
                        continue
                    self._data_buffer[symbol] = deque(maxlen=500)
                    for _, row in df.iterrows():
                        self._data_buffer[symbol].append(row.to_dict())
                    print(f"   Loaded {len(df)} candles for {symbol}", flush=True)
                except Exception as e:
                    print(f"   ERROR loading {symbol}: {e}", flush=True)
                    logger.error(f"Failed to load {symbol}: {e}")

    async def _on_kline_closed(self, kline_info: Dict) -> None:
        self._stats['klines'] += 1
        symbol_with_slash = kline_info['symbol'][:-4] + "/USDT"
        print(f"\n{'='*60}", flush=True)
        print(f"   NEW CANDLE: {symbol_with_slash} @ {kline_info['timestamp']}", flush=True)
        print(f"   O:{kline_info['open']:.6f} H:{kline_info['high']:.6f} L:{kline_info['low']:.6f} C:{kline_info['close']:.6f}", flush=True)
        print(f"   Vol:{kline_info['volume']:.2f} Trades:{kline_info['num_trades']}", flush=True)
        print(f"{'='*60}", flush=True)
        if symbol_with_slash not in self._data_buffer:
            self._data_buffer[symbol_with_slash] = deque(maxlen=500)
        self._data_buffer[symbol_with_slash].append(kline_info)
        try:
            await self._process_symbol(symbol_with_slash)
        except Exception as e:
            print(f"   ❌ PROCESS ERROR: {e}", flush=True)
            logger.error(f"Process error: {e}", exc_info=True)

    async def _process_symbol(self, symbol: str) -> None:
        if symbol not in self._data_buffer:
            print(f"   No buffer for {symbol}", flush=True)
            return

        buffer_len = len(self._data_buffer[symbol])
        if buffer_len < MIN_HISTORY_CANDLES:
            print(f"   Buffering: {buffer_len}/{MIN_HISTORY_CANDLES} candles (need more history)", flush=True)
            return

        history_df = pd.DataFrame(list(self._data_buffer[symbol]))
        if 'timestamp' not in history_df.columns:
            print(f"   ERROR: No 'timestamp' column in history", flush=True)
            return

        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp').reset_index(drop=True)

        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                     'taker_buy_base_volume', 'taker_buy_quote_volume']:
            if col in history_df.columns:
                history_df[col] = pd.to_numeric(history_df[col], errors='coerce')
        if 'num_trades' in history_df.columns:
            history_df['num_trades'] = pd.to_numeric(history_df['num_trades'], errors='coerce').fillna(0).astype(int)

        btc_df = pd.DataFrame(list(self._data_buffer.get(self._pipeline_config.btc_symbol, [])))
        if not btc_df.empty:
            btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                         'taker_buy_base_volume', 'taker_buy_quote_volume']:
                if col in btc_df.columns:
                    btc_df[col] = pd.to_numeric(btc_df[col], errors='coerce')

        try:
            features_df = self._feature_engineer.generate_all_features(
                history_df, btc_df if not btc_df.empty else None
            )
        except Exception as e:
            print(f"   ❌ FEATURE ERROR: {e}", flush=True)
            logger.error(f"Feature generation error for {symbol}: {e}", exc_info=True)
            return

        if features_df.empty:
            print(f"   WARNING: Feature generation returned empty DataFrame", flush=True)
            return

        feature_cols = [c for c in features_df.columns if c.endswith('_lag1')]
        print(f"   Features: {len(feature_cols)} columns", flush=True)

        if not feature_cols:
            print(f"   WARNING: No _lag1 features generated!", flush=True)
            return

        latest = features_df[feature_cols].iloc[-1].fillna(0)
        price = features_df['close'].iloc[-1]
        volume = features_df['volume'].iloc[-1] if 'volume' in features_df.columns else 0

        if self._expected_features:
            aligned_values = []
            for feat in self._expected_features:
                if feat in latest.index:
                    aligned_values.append(latest[feat])
                else:
                    aligned_values.append(0.0)
            X_dict = {
                'feature_names': self._expected_features,
                'feature_values': aligned_values,
            }
            missing = [f for f in self._expected_features if f not in latest.index]
            if missing:
                print(f"   NOTE: {len(missing)} features filled with 0: {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
        else:
            X_dict = {'feature_names': feature_cols, 'feature_values': latest.tolist()}

        print(f"   Running inference...", flush=True)
        t0 = time.time()
        pump_prob, dump_prob = inference_sequential(
            self._pump_model_path, self._dump_model_path, X_dict
        )
        t_ms = (time.time() - t0) * 1000
        self._stats['times'].append(t_ms)

        print(f"\n   INFERENCE: {symbol}", flush=True)
        print(f"   Time: {t_ms:.1f}ms", flush=True)
        print(f"   PUMP: {pump_prob:.2%}", flush=True)
        print(f"   DUMP: {dump_prob:.2%}", flush=True)
        print(f"   Price: {price:.6f}", flush=True)
        await self._check_alert(symbol, pump_prob, dump_prob, price, volume, latest.to_dict())

    async def _check_alert(self, symbol, pump, dump, price, vol, features):
        pump_thr = self.config.pump_alert_threshold
        dump_thr = self.config.dump_alert_threshold
        if pump >= pump_thr and pump > dump:
            sig, conf = "PUMP", pump
        elif dump >= dump_thr and dump > pump:
            sig, conf = "DUMP", dump
        else:
            print(f"   NEUTRAL (threshold {pump_thr:.0%})", flush=True)
            return
        self._stats['alerts'] += 1
        print(f"\n{'='*60}", flush=True)
        print(f"   ALERT! {'PUMP' if sig=='PUMP' else 'DUMP'}", flush=True)
        print(f"   {symbol}: {sig} @ {conf:.2%}", flush=True)
        print(f"   PUMP={pump:.2%} DUMP={dump:.2%}", flush=True)
        print(f"{'='*60}\n", flush=True)
        Path("logs").mkdir(exist_ok=True)
        alert = DualAlert(datetime.utcnow(), symbol, pump, dump, sig, conf, price, vol, features)
        with open("logs/alerts.jsonl", "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")
        if self.alert_callback:
            self.alert_callback(alert)


def example_dual_alert_handler(alert: DualAlert):
    print(f"\n   ALERT: {alert.symbol} {alert.dominant_signal} @ {alert.confidence:.2%}", flush=True)


async def run_dual_gpu_inference(config: DualGPUConfig, symbols: List[str], alert_callback=None):
    engine = DualGPUInferenceEngine(config, symbols, alert_callback)
    await engine.start()
