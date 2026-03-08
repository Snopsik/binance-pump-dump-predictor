"""
Live Inference модуль для предсказания пампов/дампов в реальном времени.

Ключевые возможности:
- Асинхронный цикл сбора данных каждую минуту
- Инкрементальное обновление признаков
- Предсказания модели с порогом вероятности
- Алерты при обнаружении аномалий

ЗАГЛУШКА: Место для WebSocket L2 Order Book Imbalance (OBI)
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path
from collections import deque

from config import PipelineConfig
from data_collector import BinanceDataCollector
from feature_engineering import FeatureEngineer, TradeFlowFeatures, VolumeAnomalyFeatures, PriceActionFeatures
from model_training import PumpDumpModel

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Типы алертов."""
    PUMP = "PUMP"
    DUMP = "DUMP"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    VOLUME_SPIKE = "VOLUME_SPIKE"


@dataclass
class Alert:
    """
    Структура алерта.

    Attributes:
        timestamp: Время алерта
        symbol: Торговый символ
        alert_type: Тип алерта
        probability: Вероятность события
        price: Текущая цена
        details: Дополнительная информация
    """
    timestamp: datetime
    symbol: str
    alert_type: AlertType
    probability: float
    price: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь для JSON."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'alert_type': self.alert_type.value,
            'probability': self.probability,
            'price': self.price,
            'details': self.details
        }

    def __str__(self) -> str:
        """Строковое представление для логирования."""
        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"🚨 {self.alert_type.value} ALERT: {self.symbol} "
            f"| Prob: {self.probability:.2%} | Price: {self.price:.4f}"
        )


class OrderBookImbalancePlaceholder:
    """
    ЗАГЛУШКА: Order Book Imbalance через WebSocket L2.

    В лайве стакан получить можно через WebSocket API Binance.
    Это позволит добавить дополнительные признаки:
    - Bid-Ask Imbalance
    - Depth-weighted price pressure
    - Large order detection

    TODO: Реализовать WebSocket подключение:

    Example implementation:
    ```python
    import websockets
    import json

    class OrderBookImbalance:
        def __init__(self, symbol: str):
            self.symbol = symbol
            self.bids = {}  # {price: quantity}
            self.asks = {}
            self.ws_url = f"wss://fstream.binance.com/ws/{symbol.lower()}@depth@100ms"

        async def connect(self):
            async with websockets.connect(self.ws_url) as ws:
                async for message in ws:
                    data = json.loads(message)
                    self._update_orderbook(data)

        def _update_orderbook(self, data: dict):
            for bid in data.get('b', []):
                price, qty = float(bid[0]), float(bid[1])
                if qty == 0:
                    self.bids.pop(price, None)
                else:
                    self.bids[price] = qty

            for ask in data.get('a', []):
                price, qty = float(ask[0]), float(ask[1])
                if qty == 0:
                    self.asks.pop(price, None)
                else:
                    self.asks[price] = qty

        def get_imbalance(self, depth: int = 10) -> float:
            top_bids = sorted(self.bids.items(), reverse=True)[:depth]
            top_asks = sorted(self.asks.items())[:depth]

            bid_volume = sum(qty for _, qty in top_bids)
            ask_volume = sum(qty for _, qty in top_asks)

            if bid_volume + ask_volume == 0:
                return 0.5

            return bid_volume / (bid_volume + ask_volume)
    ```
    """

    def __init__(self, symbol: str):
        """Инициализация заглушки OBI."""
        self.symbol = symbol
        logger.warning(
            f"OrderBookImbalance is a placeholder for {symbol}. "
            "WebSocket L2 implementation needed for live OBI features."
        )

    async def connect(self) -> None:
        """Подключение к WebSocket (заглушка)."""
        pass

    def get_imbalance(self, depth: int = 10) -> float:
        """Получить imbalance (заглушка возвращает нейтральное значение)."""
        return 0.5

    def get_depth_metrics(self) -> Dict[str, float]:
        """Получить метрики глубины (заглушка)."""
        return {
            'bid_depth': 0.0,
            'ask_depth': 0.0,
            'imbalance': 0.5
        }


class LiveDataBuffer:
    """
    Буфер для хранения последних данных в памяти.

    Поддерживает скользящее окно данных для расчета признаков.
    """

    def __init__(self, max_size: int = 500):
        """
        Инициализация буфера.

        Args:
            max_size: Максимальное количество записей
        """
        self.max_size = max_size
        self._data: Dict[str, deque] = {}

    def add(self, symbol: str, record: Dict[str, Any]) -> None:
        """
        Добавить запись в буфер.

        Args:
            symbol: Торговый символ
            record: Словарь с данными OHLCV
        """
        if symbol not in self._data:
            self._data[symbol] = deque(maxlen=self.max_size)
        self._data[symbol].append(record)

    def get_df(self, symbol: str) -> pd.DataFrame:
        """
        Получить DataFrame для символа.

        Args:
            symbol: Торговый символ

        Returns:
            DataFrame с историческими данными
        """
        if symbol not in self._data or len(self._data[symbol]) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(list(self._data[symbol]))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)

    def get_latest(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """
        Получить последние n записей.

        Args:
            symbol: Торговый символ
            n: Количество записей

        Returns:
            DataFrame с последними данными
        """
        if symbol not in self._data:
            return pd.DataFrame()

        records = list(self._data[symbol])[-n:]
        return pd.DataFrame(records)


class LiveFeatureCalculator:
    """
    Инкрементальный калькулятор признаков для live inference.

    Оптимизирован для быстрого расчета признаков на лету
    без полного пересчета всей истории.
    """

    def __init__(self, config: PipelineConfig):
        """
        Инициализация калькулятора.

        Args:
            config: Конфигурация пайплайна
        """
        self.config = config
        self.feature_config = config.features

        # Кэши для скользящих расчетов
        self._cache: Dict[str, Dict[str, Any]] = {}

    def calculate_features_incremental(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Рассчитать признаки для последней свечи инкрементально.

        Args:
            df: DataFrame с историческими данными (включая новую свечу)
            symbol: Торговый символ

        Returns:
            DataFrame с признаками для последней свечи
        """
        if len(df) < 2:
            return pd.DataFrame()

        # Берем последнюю закрытую свечу и историю
        latest = df.iloc[-1:].copy()
        history = df.iloc[:-1]

        # Минимальная история для расчетов
        min_history = max(
            self.feature_config.very_long_window,
            max(self.feature_config.volume_zscore_windows)
        )

        if len(history) < min_history:
            logger.warning(
                f"Insufficient history for {symbol}: {len(history)} < {min_history}"
            )
            # Используем что есть

        # Рассчитываем признаки
        latest = self._calculate_single_row_features(latest, history)

        return latest

    def _calculate_single_row_features(
        self,
        current: pd.DataFrame,
        history: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Рассчитать признаки для одной строки на основе истории.

        Args:
            current: Текущая свеча (1 row)
            history: Исторические данные

        Returns:
            DataFrame с признаками
        """
        result = current.copy()

        # Объединяем для расчета скользящих метрик
        combined = pd.concat([history, current], ignore_index=True)

        # === TRADE FLOW FEATURES ===
        # Taker buy ratio (используем предыдущую свечу для предотвращения leakage)
        taker_buy_ratio_prev = TradeFlowFeatures.taker_buy_ratio(combined).iloc[-2]
        result['taker_buy_ratio_lag1'] = taker_buy_ratio_prev

        # CVD
        for window in self.feature_config.cvd_windows:
            cvd = TradeFlowFeatures.cvd(combined, window).iloc[-2]
            result[f'cvd_{window}_lag1'] = cvd

            cvd_norm = TradeFlowFeatures.cvd_normalized(combined, window).iloc[-2]
            result[f'cvd_norm_{window}_lag1'] = cvd_norm

        # Delta MA ratio
        delta_ma_ratio = TradeFlowFeatures.delta_ma_ratio(combined, 10).iloc[-2]
        result['delta_ma_ratio_10_lag1'] = delta_ma_ratio

        delta_ma_ratio_30 = TradeFlowFeatures.delta_ma_ratio(combined, 30).iloc[-2]
        result['delta_ma_ratio_30_lag1'] = delta_ma_ratio_30

        # Aggressive trade size
        aggressive_size = TradeFlowFeatures.aggressive_trade_size(combined).iloc[-2]
        result['aggressive_trade_size_lag1'] = aggressive_size

        # === VOLUME ANOMALY FEATURES ===
        for window in self.feature_config.volume_zscore_windows:
            vol_zscore = VolumeAnomalyFeatures.volume_zscore(combined, window).iloc[-2]
            result[f'vol_zscore_{window}_lag1'] = vol_zscore

        # rvol_seasonal
        rvol = VolumeAnomalyFeatures.rvol_seasonal(combined).iloc[-2]
        result['rvol_seasonal_lag1'] = rvol

        # Vol acceleration
        vol_acc = VolumeAnomalyFeatures.vol_acceleration(combined).iloc[-2]
        result['vol_acceleration_lag1'] = vol_acc

        # Vol spikes
        vol_spike_20 = VolumeAnomalyFeatures.volume_spike(combined, 20, 2.0).iloc[-2]
        result['vol_spike_20_lag1'] = vol_spike_20

        vol_spike_60 = VolumeAnomalyFeatures.volume_spike(combined, 60, 2.5).iloc[-2]
        result['vol_spike_60_lag1'] = vol_spike_60

        # === PRICE ACTION FEATURES ===
        # Log return
        log_return = PriceActionFeatures.log_returns(combined).iloc[-2]
        result['log_return_lag1'] = log_return

        # Parkinson vol
        parkinson = PriceActionFeatures.parkinson_vol(combined, 20).iloc[-2]
        result['parkinson_vol_20_lag1'] = parkinson

        # ATR
        atr = PriceActionFeatures.atr(combined, 14).iloc[-2]
        result['atr_14_lag1'] = atr

        # Breakouts
        breakout_upper = PriceActionFeatures.breakout_upper(combined, 20).iloc[-2]
        result['breakout_upper_20_lag1'] = breakout_upper

        breakout_lower = PriceActionFeatures.breakout_lower(combined, 20).iloc[-2]
        result['breakout_lower_20_lag1'] = breakout_lower

        # Wick ratios
        upper_wick = PriceActionFeatures.upper_wick_ratio(combined).iloc[-2]
        result['upper_wick_ratio_lag1'] = upper_wick

        lower_wick = PriceActionFeatures.lower_wick_ratio(combined).iloc[-2]
        result['lower_wick_ratio_lag1'] = lower_wick

        # Candle body ratio
        body_ratio = PriceActionFeatures.candle_body_ratio(combined).iloc[-2]
        result['candle_body_ratio_lag1'] = body_ratio

        result['rsi_14_lag1'] = PriceActionFeatures.rsi(combined, 14).iloc[-2]
        result['rsi_7_lag1'] = PriceActionFeatures.rsi(combined, 7).iloc[-2]
        result['bb_width_20_lag1'] = PriceActionFeatures.bollinger_band_width(combined, 20).iloc[-2]
        result['bb_pct_b_20_lag1'] = PriceActionFeatures.bollinger_pct_b(combined, 20).iloc[-2]
        result['vwap_dev_60_lag1'] = PriceActionFeatures.vwap_deviation(combined, 60).iloc[-2]
        result['roc_5_lag1'] = PriceActionFeatures.rate_of_change(combined, 5).iloc[-2]
        result['roc_10_lag1'] = PriceActionFeatures.rate_of_change(combined, 10).iloc[-2]
        result['roc_30_lag1'] = PriceActionFeatures.rate_of_change(combined, 30).iloc[-2]
        result['trade_count_zscore_60_lag1'] = PriceActionFeatures.trade_count_zscore(combined, 60).iloc[-2]

        return result

    def add_btc_correlation_features(
        self,
        df: pd.DataFrame,
        btc_history: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Добавить признаки корреляции с BTC.

        Args:
            df: DataFrame с признаками альткоина
            btc_history: Исторические данные BTC

        Returns:
            DataFrame с добавленными признаками
        """
        if btc_history.empty:
            return df

        # Рассчитываем доходности BTC
        btc_returns = PriceActionFeatures.log_returns(btc_history)

        # Для альткоина тоже нужны доходности (берем из истории)
        # Это упрощенный расчет - в production нужно использовать полную историю

        # Заполняем NaN
        df['btc_corr_30_lag1'] = df.get('btc_corr_30_lag1', 0.5)
        df['btc_corr_drop_lag1'] = df.get('btc_corr_drop_lag1', 0.0)
        df['relative_strength_30_lag1'] = df.get('relative_strength_30_lag1', 0.0)
        df['atr_norm_spread_lag1'] = df.get('atr_norm_spread_lag1', 0.0)

        return df


class LiveInferenceEngine:
    """
    Главный класс для live inference.

    Объединяет сбор данных, расчет признаков и предсказания модели.
    """

    def __init__(
        self,
        config: PipelineConfig,
        model: PumpDumpModel,
        symbols: List[str],
        alert_callback: Optional[Callable[[Alert], None]] = None
    ):
        """
        Инициализация inference engine.

        Args:
            config: Конфигурация пайплайна
            model: Обученная модель
            symbols: Список символов для мониторинга
            alert_callback: Callback функция для обработки алертов
        """
        self.config = config
        self.model = model
        self.symbols = symbols
        self.alert_callback = alert_callback

        self.buffer = LiveDataBuffer(max_size=500)
        self.feature_calculator = LiveFeatureCalculator(config)

        # Порог вероятности для алертов
        self.alert_threshold = config.model_params.alert_threshold

        # Состояние
        self._running = False
        self._last_update: Dict[str, datetime] = {}

        # Order Book Imbalance (заглушка)
        self._obi: Dict[str, OrderBookImbalancePlaceholder] = {}

        # Символ BTC для режима рынка
        self.btc_symbol = config.btc_symbol

    async def start(self) -> None:
        """
        Запустить live inference цикл.

        Выполняется каждую минуту после закрытия свечи.
        """
        self._running = True

        logger.info(f"Starting Live Inference Engine for {len(self.symbols)} symbols")
        logger.info(f"Alert threshold: {self.alert_threshold:.0%}")

        async with BinanceDataCollector(self.config) as collector:
            # Загружаем начальную историю
            await self._load_initial_history(collector)

            # Основной цикл
            while self._running:
                try:
                    await self._inference_cycle(collector)
                except Exception as e:
                    logger.error(f"Error in inference cycle: {e}", exc_info=True)

                # Ждем до следующей минуты
                await self._wait_for_next_minute()

    async def stop(self) -> None:
        """Остановить inference цикл."""
        self._running = False
        logger.info("Live Inference Engine stopped")

    async def _load_initial_history(self, collector: BinanceDataCollector) -> None:
        """
        Загрузить начальную историю для всех символов.

        Args:
            collector: Сборщик данных
        """
        logger.info("Loading initial history...")

        all_symbols = self.symbols + [self.btc_symbol]

        for symbol in all_symbols:
            try:
                # Загружаем последние 500 свечей
                df = await collector.fetch_ohlcv(symbol, limit=500)

                if df.empty:
                    logger.warning(f"No data loaded for {symbol}")
                    continue

                # Добавляем в буфер
                for _, row in df.iterrows():
                    self.buffer.add(symbol, row.to_dict())

                logger.info(f"Loaded {len(df)} candles for {symbol}")

            except Exception as e:
                logger.error(f"Failed to load history for {symbol}: {e}")

        logger.info("Initial history loaded")

    async def _inference_cycle(self, collector: BinanceDataCollector) -> None:
        """
        Выполнить один цикл inference.

        Args:
            collector: Сборщик данных
        """
        current_time = datetime.utcnow()

        logger.info(f"\n{'='*60}")
        logger.info(f"Inference cycle at {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("="*60)

        # Обновляем данные для всех символов
        for symbol in self.symbols + [self.btc_symbol]:
            try:
                # Получаем последнюю закрытую свечу
                latest_df = await collector.get_latest_candle(symbol)

                if latest_df is None or latest_df.empty:
                    logger.warning(f"No new data for {symbol}")
                    continue

                # Проверяем, что это новая свеча
                new_timestamp = latest_df['timestamp'].iloc[0]
                last_timestamp = self._last_update.get(symbol)

                if last_timestamp and new_timestamp <= last_timestamp:
                    continue  # Уже обработали эту свечу

                # Добавляем в буфер
                for _, row in latest_df.iterrows():
                    self.buffer.add(symbol, row.to_dict())

                self._last_update[symbol] = new_timestamp

                # Для BTC только обновляем буфер
                if symbol == self.btc_symbol:
                    continue

                # Рассчитываем признаки и делаем предсказание
                await self._process_symbol(symbol)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    async def _process_symbol(self, symbol: str) -> None:
        """
        Обработать символ: рассчитать признаки и сделать предсказание.

        Args:
            symbol: Торговый символ
        """
        # Получаем историю
        history = self.buffer.get_df(symbol)
        btc_history = self.buffer.get_df(self.btc_symbol)

        if history.empty or len(history) < 30:
            logger.warning(f"Insufficient data for {symbol}: {len(history)} candles")
            return

        # Рассчитываем признаки
        features_df = self.feature_calculator.calculate_features_incremental(
            history, symbol
        )

        if features_df.empty:
            return

        # Добавляем признаки корреляции с BTC
        features_df = self.feature_calculator.add_btc_correlation_features(
            features_df, btc_history
        )

        # Подготавливаем признаки для модели
        feature_cols = [col for col in features_df.columns
                       if col.endswith('_lag1')]

        if not feature_cols:
            logger.warning(f"No feature columns for {symbol}")
            return

        X = features_df[feature_cols]

        # Заполняем NaN
        X = X.fillna(0)

        # Делаем предсказание
        try:
            proba = self.model.predict_proba(X)
            pump_prob = proba[0, 2]  # Класс 2 = памп
            dump_prob = proba[0, 0]  # Класс 0 = дамп

            logger.info(
                f"{symbol}: Pump={pump_prob:.2%}, Dump={dump_prob:.2%}, "
                f"Price={features_df['close'].iloc[0]:.6f}"
            )

            # Проверяем пороги и генерируем алерты
            await self._check_and_alert(
                symbol=symbol,
                pump_prob=pump_prob,
                dump_prob=dump_prob,
                price=features_df['close'].iloc[0],
                features=features_df
            )

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")

    async def _check_and_alert(
        self,
        symbol: str,
        pump_prob: float,
        dump_prob: float,
        price: float,
        features: pd.DataFrame
    ) -> None:
        """
        Проверить пороги и сгенерировать алерт.

        Args:
            symbol: Торговый символ
            pump_prob: Вероятность пампа
            dump_prob: Вероятность дампа
            price: Текущая цена
            features: DataFrame с признаками
        """
        alert = None

        # Проверяем памп
        if pump_prob >= self.alert_threshold:
            alert = Alert(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                alert_type=AlertType.PUMP,
                probability=pump_prob,
                price=price,
                details={
                    'dump_probability': dump_prob,
                    'volume_zscore': features.get('vol_zscore_60_lag1', [0]).iloc[0],
                    'cvd_30': features.get('cvd_30_lag1', [0]).iloc[0],
                    'breakout_upper': features.get('breakout_upper_20_lag1', [0]).iloc[0],
                }
            )

        # Проверяем дамп
        elif dump_prob >= self.alert_threshold:
            alert = Alert(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                alert_type=AlertType.DUMP,
                probability=dump_prob,
                price=price,
                details={
                    'pump_probability': pump_prob,
                    'volume_zscore': features.get('vol_zscore_60_lag1', [0]).iloc[0],
                    'cvd_30': features.get('cvd_30_lag1', [0]).iloc[0],
                    'breakout_lower': features.get('breakout_lower_20_lag1', [0]).iloc[0],
                }
            )

        if alert:
            # Логируем алерт
            logger.warning(str(alert))

            # Сохраняем в файл
            self._save_alert(alert)

            # Вызываем callback если есть
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    def _save_alert(self, alert: Alert) -> None:
        """
        Сохранить алерт в файл.

        Args:
            alert: Объект алерта
        """
        alert_file = Path("logs/alerts.jsonl")
        alert_file.parent.mkdir(parents=True, exist_ok=True)

        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert.to_dict()) + '\n')

    async def _wait_for_next_minute(self) -> None:
        """Ждать до начала следующей минуты."""
        now = datetime.utcnow()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        wait_seconds = (next_minute - now).total_seconds()

        if wait_seconds > 0:
            logger.info(f"Waiting {wait_seconds:.1f}s until next minute...")
            await asyncio.sleep(wait_seconds)


async def run_live_inference(
    config: PipelineConfig,
    model_path: str,
    symbols: List[str],
    alert_callback: Optional[Callable[[Alert], None]] = None
) -> None:
    """
    Запустить live inference.

    Args:
        config: Конфигурация пайплайна
        model_path: Путь к сохраненной модели
        symbols: Список символов для мониторинга
        alert_callback: Callback для обработки алертов
    """
    # Загружаем модель
    model = PumpDumpModel(config)
    model.load(model_path)

    # Создаем и запускаем engine
    engine = LiveInferenceEngine(
        config=config,
        model=model,
        symbols=symbols,
        alert_callback=alert_callback
    )

    await engine.start()


def example_alert_handler(alert: Alert) -> None:
    """
    Пример обработчика алертов.

    В production здесь может быть:
    - Отправка в Telegram/Discord
    - Выполнение trade сигнала
    - Запись в базу данных

    Args:
        alert: Объект алерта
    """
    print(f"\n{'🚨'*20}")
    print(f"ALERT RECEIVED!")
    print(f"Symbol: {alert.symbol}")
    print(f"Type: {alert.alert_type.value}")
    print(f"Probability: {alert.probability:.2%}")
    print(f"Price: {alert.price}")
    print(f"{'🚨'*20}\n")