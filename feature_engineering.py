"""
Advanced Quant Feature Engineering для предсказания пампов/дампов.

Генерирует следующие группы признаков:
1. Trade Flow - анализ агрессивных покупок/продаж
2. Volume Anomaly - аномалии объема
3. Price Action - ценовое действие и волатильность
4. Market Regime - режим рынка и корреляция с BTC

КРИТИЧЕСКИ ВАЖНО: Все признаки сдвигаются на 1 период назад (.shift(1))
перед объединением с таргетом для предотвращения Data Leakage.

OPTIMIZED: Uses numba JIT compilation for heavy calculations (~10-50x faster).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import warnings

# Try to import numba for JIT compilation (optional but recommended for performance)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

from config import PipelineConfig, FeatureConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class FeatureGroups:
    """Названия групп признаков для удобства фильтрации."""
    TRADE_FLOW: str = "trade_flow"
    VOLUME_ANOMALY: str = "volume_anomaly"
    PRICE_ACTION: str = "price_action"
    MARKET_REGIME: str = "market_regime"
    RAW: str = "raw"


# =============================================================================
# NUMBA-OPTIMIZED FUNCTIONS FOR PERFORMANCE (~10-50x faster than pure Python)
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _fast_rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling sum using numba JIT compilation."""
    n = len(arr)
    result = np.empty(n)
    result[:window] = np.nan
    
    for i in prange(window, n):
        result[i] = np.nansum(arr[i-window:i])
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def _fast_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using numba JIT compilation."""
    n = len(arr)
    result = np.empty(n)
    result[:window] = np.nan
    
    for i in prange(window, n):
        valid_count = 0
        total = 0.0
        for j in range(i - window, i):
            if not np.isnan(arr[j]):
                total += arr[j]
                valid_count += 1
        result[i] = total / valid_count if valid_count > 0 else np.nan
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def _fast_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation using numba JIT compilation."""
    n = len(arr)
    result = np.empty(n)
    result[:window] = np.nan
    
    for i in prange(window, n):
        valid_values = arr[i-window:i]
        valid_values = valid_values[~np.isnan(valid_values)]
        if len(valid_values) > 1:
            mean = np.mean(valid_values)
            result[i] = np.sqrt(np.mean((valid_values - mean) ** 2))
        else:
            result[i] = np.nan
    
    return result


@jit(nopython=True, cache=True)
def _fast_parkinson_volatility(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Fast Parkinson volatility calculation.
    Uses high-low range for volatility estimation (more efficient than close-to-close).
    """
    n = len(high)
    result = np.empty(n)
    result[:window] = np.nan
    
    log_hl_sq = np.log(high / low) ** 2
    k = 1.0 / (4.0 * window * np.log(2))
    
    for i in range(window, n):
        result[i] = np.sqrt(k * np.nansum(log_hl_sq[i-window:i]))
    
    return result


@jit(nopython=True, cache=True)
def _fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Fast Average True Range calculation."""
    n = len(high)
    result = np.empty(n)
    result[:window] = np.nan
    
    # Calculate True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Rolling mean of TR
    for i in range(window, n):
        result[i] = np.nanmean(tr[i-window:i])
    
    return result


@jit(nopython=True, cache=True)
def _fast_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling z-score calculation."""
    n = len(arr)
    result = np.empty(n)
    result[:window] = np.nan
    
    for i in range(window, n):
        window_data = arr[i-window:i]
        valid_mask = ~np.isnan(window_data)
        valid_data = window_data[valid_mask]
        
        if len(valid_data) > 1:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            if std > 0:
                result[i] = (arr[i] - mean) / std
            else:
                result[i] = 0.0
        else:
            result[i] = np.nan
    
    return result


class TradeFlowFeatures:
    """
    Генератор признаков Trade Flow.

    Анализирует агрессивную покупку/продажу на основе taker volume.

    Ключевые метрики:
    - taker_buy_ratio: Доля агрессивных покупок в общем объеме
    - CVD (Cumulative Volume Delta): Накопительная дельта объема
    - delta_ma_ratio: Отношение дельты к скользящей средней
    """

    @staticmethod
    def taker_buy_ratio(df: pd.DataFrame) -> pd.Series:
        """
        Рассчитать долю агрессивных покупок в общем объеме.

        Taker Buy Ratio = taker_buy_base_volume / volume

        Показывает, какая часть объема была инициирована агрессивными покупателями.
        Высокие значения (>0.7) указывают на сильное покупательское давление.
        Низкие значения (<0.3) указывают на давление продавцов.

        Args:
            df: DataFrame с колонками 'taker_buy_base_volume' и 'volume'

        Returns:
            Series с taker_buy_ratio
        """
        return df['taker_buy_base_volume'] / df['volume'].replace(0, np.nan)

    @staticmethod
    def taker_buy_quote_ratio(df: pd.DataFrame) -> pd.Series:
        """
        Рассчитать долю агрессивных покупок в котируемой валюте.

        Полезно для анализа в USDT эквиваленте.

        Args:
            df: DataFrame с колонками 'taker_buy_quote_volume' и 'quote_volume'

        Returns:
            Series с taker_buy_quote_ratio
        """
        return df['taker_buy_quote_volume'] / df['quote_volume'].replace(0, np.nan)

    @staticmethod
    def calculate_delta(df: pd.DataFrame) -> pd.Series:
        """
        Рассчитать дельту объема (разница между покупками и продажами).

        Delta = taker_buy_base_volume - (volume - taker_buy_base_volume)
              = 2 * taker_buy_base_volume - volume

        Положительная дельта = больше агрессивных покупателей
        Отрицательная дельта = больше агрессивных продавцов

        Args:
            df: DataFrame с колонками 'taker_buy_base_volume' и 'volume'

        Returns:
            Series с delta
        """
        return 2 * df['taker_buy_base_volume'] - df['volume']

    @staticmethod
    def cvd(df: pd.DataFrame, window: int) -> pd.Series:
        """
        Cumulative Volume Delta - накопительная дельта за окно.

        Показывает совокупный дисбаланс между покупками и продажами.
        Растущий CVD указывает на накопление покупателями.
        Падающий CVD указывает на накопление продавцами.

        Args:
            df: DataFrame с данными
            window: Размер окна для накопления

        Returns:
            Series с CVD
        """
        delta = TradeFlowFeatures.calculate_delta(df)
        return delta.rolling(window=window, min_periods=1).sum()

    @staticmethod
    def cvd_normalized(df: pd.DataFrame, window: int) -> pd.Series:
        """
        Нормализованный CVD (деленный на общий объем за окно).

        Позволяет сравнивать CVD между разными активами с разным объемом.

        Args:
            df: DataFrame с данными
            window: Размер окна

        Returns:
            Series с нормализованным CVD
        """
        delta = TradeFlowFeatures.calculate_delta(df)
        cvd = delta.rolling(window=window, min_periods=1).sum()
        volume_sum = df['volume'].rolling(window=window, min_periods=1).sum()
        return cvd / volume_sum.replace(0, np.nan)

    @staticmethod
    def delta_ma_ratio(df: pd.DataFrame, window: int) -> pd.Series:
        """
        Отношение текущей дельты к скользящей средней дельты.

        Показывает, насколько текущий дисбаланс отличается от типичного.

        Значения > 2: аномально высокая покупательская активность
        Значения < 0.5: аномально низкая покупательская активность

        Args:
            df: DataFrame с данными
            window: Размер окна для MA

        Returns:
            Series с delta_ma_ratio
        """
        delta = TradeFlowFeatures.calculate_delta(df)
        delta_ma = delta.rolling(window=window, min_periods=1).mean()
        return delta / delta_ma.replace(0, np.nan)

    @staticmethod
    def aggressive_trade_size(df: pd.DataFrame) -> pd.Series:
        """
        Средний размер агрессивной сделки.

        aggressive_size = taker_buy_base_volume / num_trades

        Рост среднего размера может указывать на присутствие крупных игроков.

        Args:
            df: DataFrame с данными

        Returns:
            Series с средним размером агрессивной сделки
        """
        return df['taker_buy_base_volume'] / df['num_trades'].replace(0, np.nan)


class VolumeAnomalyFeatures:
    """
    Генератор признаков Volume Anomaly.

    Детектирует аномалии в объеме торгов.

    Ключевые метрики:
    - Z-score объема на разных окнах
    - Относительный объем (rvol_seasonal)
    - Ускорение объема
    """

    @staticmethod
    def volume_zscore(df: pd.DataFrame, window: int) -> pd.Series:
        """
        Z-score объема относительно скользящего окна.

        Показывает, на сколько стандартных отклонений текущий объем
        отличается от среднего за окно.

        Z-score > 3: аномально высокий объем
        Z-score < -3: аномально низкий объем

        Args:
            df: DataFrame с колонкой 'volume'
            window: Размер окна для расчета статистик

        Returns:
            Series с z-score объема
        """
        if NUMBA_AVAILABLE:
            # Use numba-optimized version
            volume = df['volume'].values.astype(np.float64)
            result = _fast_zscore(volume, window)
            return pd.Series(result, index=df.index)
        else:
            # Fallback to pure pandas
            rolling_mean = df['volume'].rolling(window=window, min_periods=1).mean()
            rolling_std = df['volume'].rolling(window=window, min_periods=1).std()
            return (df['volume'] - rolling_mean) / rolling_std.replace(0, np.nan)

    @staticmethod
    def rvol_seasonal(df: pd.DataFrame, window: int = 1440) -> pd.Series:
        """
        Сезонный относительный объем.

        Сравнивает текущий объем с объемом в то же время предыдущих дней.
        Для 1m таймфрейма window=1440 (один день).

        Args:
            df: DataFrame с колонкой 'volume'
            window: Период сезонности (1440 минут = 1 день для 1m TF)

        Returns:
            Series с сезонным относительным объемом
        """
        # Текущий объем / средний объем в то же время прошлых дней
        # Сдвигаем на window и берем среднее
        volume_same_time = df['volume'].shift(window)
        volume_ma = volume_same_time.rolling(window=window, min_periods=1).mean()
        return df['volume'] / volume_ma.replace(0, np.nan)

    @staticmethod
    def vol_acceleration(df: pd.DataFrame, short_window: int = 3, long_window: int = 10) -> pd.Series:
        """
        Ускорение объема.

        Показывает, ускоряется или замедляется рост объема.
        Положительные значения = объем ускоряется.
        Отрицательные значения = объем замедляется.

        Args:
            df: DataFrame с колонкой 'volume'
            short_window: Короткое окно для текущей динамики
            long_window: Длинное окно для базовой динамики

        Returns:
            Series с ускорением объема
        """
        vol_short_ma = df['volume'].rolling(window=short_window, min_periods=1).mean()
        vol_long_ma = df['volume'].rolling(window=long_window, min_periods=1).mean()

        # Производная (изменение) относительного объема
        vol_ratio = vol_short_ma / vol_long_ma.replace(0, np.nan)
        return vol_ratio.diff()

    @staticmethod
    def volume_spike(df: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.Series:
        """
        Бинарный признак всплеска объема.

        volume_spike = 1 if volume > mean + threshold * std else 0

        Args:
            df: DataFrame с колонкой 'volume'
            window: Размер окна
            threshold: Порог в стандартных отклонениях

        Returns:
            Series с бинарным признаком
        """
        rolling_mean = df['volume'].rolling(window=window, min_periods=1).mean()
        rolling_std = df['volume'].rolling(window=window, min_periods=1).std()
        threshold_volume = rolling_mean + threshold * rolling_std
        return (df['volume'] > threshold_volume).astype(int)


class PriceActionFeatures:
    """
    Генератор признаков Price Action.

    Анализирует ценовое движение и волатильность.

    Ключевые метрики:
    - Волатильность Паркинсона
    - Пробои локальных максимумов
    - Размер фитилей (wicks)
    """

    @staticmethod
    def log_returns(df: pd.DataFrame) -> pd.Series:
        """
        Логарифмические доходности.

        log_return = ln(close / close_prev)

        Args:
            df: DataFrame с колонкой 'close'

        Returns:
            Series с логарифмическими доходностями
        """
        return np.log(df['close'] / df['close'].shift(1))

    @staticmethod
    def parkinson_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Parkinson volatility - optimized with numba JIT compilation.
        Uses high-low range for volatility estimation (more efficient than close-to-close).
        """
        if NUMBA_AVAILABLE:
            # Use numba-optimized version (much faster for large datasets)
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            result = _fast_parkinson_volatility(high, low, window)
            return pd.Series(result, index=df.index)
        else:
            # Fallback to pure pandas
            hl_ratio = df['high'] / df['low']
            log_hl = np.log(hl_ratio)
            squared_log_hl = log_hl ** 2
            k = 1.0 / (4.0 * window * np.log(2))
            rolling_sum = squared_log_hl.rolling(window=window, min_periods=1).sum()
            parkinson_var = k * rolling_sum
            return parkinson_var ** 0.5

    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Average True Range - optimized with numba JIT compilation.
        Measures market volatility by decomposing the entire range of an asset price.
        """
        if NUMBA_AVAILABLE:
            # Use numba-optimized version (much faster for large datasets)
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            result = _fast_atr(high, low, close, window)
            return pd.Series(result, index=df.index)
        else:
            # Fallback to pure pandas
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def breakout_upper(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Признак пробоя верхнего уровня сопротивления.

        breakout = 1 if close > rolling_max(shift(1)) else 0

        Важно: используем shift(1) чтобы не смотреть в будущее!

        Args:
            df: DataFrame с колонками 'close', 'high'
            window: Размер окна для определения уровня

        Returns:
            Series с бинарным признаком пробоя
        """
        rolling_max = df['high'].shift(1).rolling(window=window, min_periods=1).max()
        return (df['close'] > rolling_max).astype(int)

    @staticmethod
    def breakout_lower(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Признак пробоя нижнего уровня поддержки.

        breakout = 1 if close < rolling_min(shift(1)) else 0

        Args:
            df: DataFrame с колонками 'close', 'low'
            window: Размер окна для определения уровня

        Returns:
            Series с бинарным признаком пробоя
        """
        rolling_min = df['low'].shift(1).rolling(window=window, min_periods=1).min()
        return (df['close'] < rolling_min).astype(int)

    @staticmethod
    def upper_wick_ratio(df: pd.DataFrame) -> pd.Series:
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        body = (df['close'] - df['open']).abs()  # ← .abs() вместо np.abs()
        return upper_wick / body.replace(0, np.nan)

    @staticmethod
    def lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        body = (df['close'] - df['open']).abs()  # ← .abs() вместо np.abs()
        return lower_wick / body.replace(0, np.nan)

    @staticmethod
    def candle_body_ratio(df: pd.DataFrame) -> pd.Series:
        body = (df['close'] - df['open']).abs()  # ← .abs() вместо np.abs()
        total_range = df['high'] - df['low']
        return body / total_range.replace(0, np.nan)

    @staticmethod
    def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_band_width(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
        sma = df['close'].rolling(window=window, min_periods=1).mean()
        std = df['close'].rolling(window=window, min_periods=1).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return (upper - lower) / sma.replace(0, np.nan)

    @staticmethod
    def bollinger_pct_b(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
        sma = df['close'].rolling(window=window, min_periods=1).mean()
        std = df['close'].rolling(window=window, min_periods=1).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        band_range = (upper - lower).replace(0, np.nan)
        return (df['close'] - lower) / band_range

    @staticmethod
    def vwap_deviation(df: pd.DataFrame, window: int = 60) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_vol = (typical_price * df['volume']).rolling(window=window, min_periods=1).sum()
        cumulative_vol = df['volume'].rolling(window=window, min_periods=1).sum()
        vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
        return (df['close'] - vwap) / vwap.replace(0, np.nan)

    @staticmethod
    def rate_of_change(df: pd.DataFrame, window: int = 10) -> pd.Series:
        return df['close'].pct_change(periods=window)

    @staticmethod
    def trade_count_zscore(df: pd.DataFrame, window: int = 60) -> pd.Series:
        if 'num_trades' not in df.columns:
            return pd.Series(0.0, index=df.index)
        trades = df['num_trades'].astype(float)
        rolling_mean = trades.rolling(window=window, min_periods=1).mean()
        rolling_std = trades.rolling(window=window, min_periods=1).std()
        return (trades - rolling_mean) / rolling_std.replace(0, np.nan)


class MarketRegimeFeatures:
    """
    Генератор признаков Market Regime.

    Анализирует режим рынка и корреляции.

    Ключевые метрики:
    - Скользящая корреляция с BTC
    - Падение корреляции (возможный знак пампа)
    - ATR-нормализованный спред
    """

    @staticmethod
    def rolling_correlation(
        returns_a: pd.Series,
        returns_b: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Скользящая корреляция между двумя сериями доходностей.

        Корреляция с BTC показывает, насколько альткоин
        следует общему движению рынка.

        Падение корреляции часто предшествует пампу -
        актив начинает двигаться независимо.

        Args:
            returns_a: Доходности первого актива
            returns_b: Доходности второго актива (например, BTC)
            window: Размер окна для корреляции

        Returns:
            Series с скользящей корреляцией
        """
        return returns_a.rolling(window=window, min_periods=1).corr(returns_b)

    @staticmethod
    def correlation_drop(
        btc_corr: pd.Series,
        threshold_window: int = 30
    ) -> pd.Series:
        """
        Падение корреляции с BTC.

        Измеряет, насколько текущая корреляция ниже средней.
        Падение корреляции - важный предиктор пампа, так как
        актив "отвязывается" от общего движения рынка.

        Args:
            btc_corr: Серия скользящей корреляции с BTC
            threshold_window: Окно для расчета базовой корреляции

        Returns:
            Series с падением корреляции
        """
        baseline_corr = btc_corr.rolling(window=threshold_window, min_periods=1).mean()
        return baseline_corr - btc_corr

    @staticmethod
    def relative_strength(
        alt_returns: pd.Series,
        btc_returns: pd.Series,
        window: int = 30
    ) -> pd.Series:
        """
        Относительная сила альткоина к BTC.

        Показывает, насколько альткоин переигрывает или
        недоигрывает BTC на данном окне.

        Args:
            alt_returns: Доходности альткоина
            btc_returns: Доходности BTC
            window: Размер окна

        Returns:
            Series с относительной силой
        """
        alt_cum = alt_returns.rolling(window=window, min_periods=1).sum()
        btc_cum = btc_returns.rolling(window=window, min_periods=1).sum()
        return alt_cum - btc_cum

    @staticmethod
    def atr_normalized_spread(
        alt_atr: pd.Series,
        alt_close: pd.Series,
        btc_atr: pd.Series,
        btc_close: pd.Series
    ) -> pd.Series:
        """
        ATR-нормализованный спред между активами.

        Показывает разницу в волатильности между активами,
        нормализованную по цене.

        Args:
            alt_atr: ATR альткоина
            alt_close: Close альткоина
            btc_atr: ATR BTC
            btc_close: Close BTC

        Returns:
            Series с нормализованным спредом
        """
        alt_vol_ratio = alt_atr / alt_close
        btc_vol_ratio = btc_atr / btc_close
        return alt_vol_ratio - btc_vol_ratio

class OIFeatures:
    """Open Interest + Funding Rate features — главные сигналы для pump detection."""

    @staticmethod
    def merge_oi_into_ohlcv(df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge OI (15m) в OHLCV (1m) через forward-fill.
        OI обновляется раз в 15 минут — заполняем вперёд.
        """
        if oi_df.empty:
            df["oi_value"] = np.nan
            df["oi_coins"] = np.nan
            return df
        
        oi = oi_df.copy()
        oi["timestamp"] = pd.to_datetime(oi["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # merge_asof — берёт последнее известное OI на момент каждой свечи
        merged = pd.merge_asof(
            df.sort_values("timestamp"),
            oi[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        df["oi_value"] = merged["sumOpenInterestValue"].values   # OI в USDT
        df["oi_coins"] = merged["sumOpenInterest"].values        # OI в монетах
        return df

    @staticmethod
    def merge_funding_into_ohlcv(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        """Merge funding rate (каждые 8ч) в OHLCV через forward-fill."""
        if funding_df.empty:
            df["funding_rate"] = np.nan
            return df
        
        fr = funding_df.copy()
        fr["timestamp"] = pd.to_datetime(fr["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        merged = pd.merge_asof(
            df.sort_values("timestamp"),
            fr[["timestamp", "fundingRate"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        df["funding_rate"] = merged["fundingRate"].values
        return df

    @staticmethod
    def add_oi_features(df: pd.DataFrame) -> pd.DataFrame:
        """Вычислить все OI-фичи из колонок oi_value и funding_rate."""
        if "oi_value" not in df.columns or df["oi_value"].isna().all():
            # Заглушки если нет данных — не ломаем пайплайн
            for col in ["oi_pct_change_15", "oi_pct_change_60", "oi_pct_change_240",
                        "oi_zscore_60", "oi_acceleration",
                        "funding_rate", "funding_extreme", "funding_squeeze_signal",
                        "oi_price_divergence", "oi_volume_ratio"]:
                df[col] = 0.0
            return df

        oi = df["oi_value"]

        # ── OI momentum (% изменение за N свечей 1m) ─────────────────────────
        df["oi_pct_change_15"]  = oi.pct_change(15)   * 100   # 15 минут
        df["oi_pct_change_60"]  = oi.pct_change(60)   * 100   # 1 час
        df["oi_pct_change_240"] = oi.pct_change(240)  * 100   # 4 часа

        # ── OI z-score (аномалия роста) ──────────────────────────────────────
        oi_mean = oi.rolling(60, min_periods=10).mean()
        oi_std  = oi.rolling(60, min_periods=10).std().replace(0, np.nan)
        df["oi_zscore_60"] = (oi - oi_mean) / oi_std

        # ── OI acceleration (скорость роста ускоряется?) ─────────────────────
        df["oi_acceleration"] = df["oi_pct_change_15"].diff(15)

        # ── Funding Rate features ─────────────────────────────────────────────
        if "funding_rate" in df.columns and not df["funding_rate"].isna().all():
            fr = df["funding_rate"]
            # Экстремальный funding (>0.1% или < -0.1%) = аномалия
            df["funding_extreme"] = (fr.abs() > 0.001).astype(int)
            # Negative funding + рост OI = short squeeze signal!
            oi_rising = (df["oi_pct_change_15"] > 2.0).astype(int)
            df["funding_squeeze_signal"] = ((fr < -0.0005) & (oi_rising == 1)).astype(int)
        else:
            df["funding_extreme"] = 0
            df["funding_squeeze_signal"] = 0

        # ── OI / Price divergence (OI растёт, цена стоит = накопление) ───────
        price_pct = df["close"].pct_change(15) * 100
        df["oi_price_divergence"] = df["oi_pct_change_15"] - price_pct

        # ── OI / Volume ratio ─────────────────────────────────────────────────
        vol_ma = df["volume"].rolling(15, min_periods=5).mean().replace(0, np.nan)
        df["oi_volume_ratio"] = oi / (vol_ma * df["close"]).replace(0, np.nan)

        return df

class FeatureEngineer:
    """
    Главный класс для генерации всех признаков.

    Объединяет все группы признаков и обеспечивает
    правильную обработку временных рядов.

    КРИТИЧЕСКИ ВАЖНО: Все признаки сдвигаются на 1 период
    для предотвращения Data Leakage.
    """

    def __init__(self, config: PipelineConfig):
        """
        Инициализация генератора признаков.

        Args:
            config: Конфигурация пайплайна
        """
        self.config = config
        self.feature_config = config.features

        # Хранение сгенерированных признаков
        self.feature_names: Dict[str, List[str]] = {
            FeatureGroups.TRADE_FLOW: [],
            FeatureGroups.VOLUME_ANOMALY: [],
            FeatureGroups.PRICE_ACTION: [],
            FeatureGroups.MARKET_REGIME: [],
            FeatureGroups.RAW: []
        }

    def generate_all_features(
        self,
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        oi_df: Optional[pd.DataFrame] = None,       # ← добавить
        funding_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Сгенерировать все признаки для одного символа.

        Args:
            df: DataFrame с OHLCV данными для символа
            btc_df: DataFrame с OHLCV данными для BTC (опционально)

        Returns:
            DataFrame со всеми признаками
        """
        df = df.copy()

        # Сортируем по времени
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Генерируем признаки по группам
        df = self._generate_trade_flow_features(df)
        df = self._generate_volume_anomaly_features(df)
        df = self._generate_price_action_features(df)
        if btc_df is not None:
            df = self._generate_market_regime_features(df, btc_df)

        # OI фичи
        df = self._generate_oi_features(
            df,
            oi_df=oi_df,
            funding_df=funding_df,
        )

        df = self._apply_shift_for_preventing_leakage(df)
        return df

    def _generate_trade_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация Trade Flow признаков.

        Args:
            df: DataFrame с исходными данными

        Returns:
            DataFrame с добавленными Trade Flow признаками
        """
        # Taker buy ratio
        df['taker_buy_ratio'] = TradeFlowFeatures.taker_buy_ratio(df)

        # Taker buy quote ratio
        if 'taker_buy_quote_volume' in df.columns and 'quote_volume' in df.columns:
            df['taker_buy_quote_ratio'] = TradeFlowFeatures.taker_buy_quote_ratio(df)

        # Delta и CVD для разных окон
        for window in self.feature_config.cvd_windows:
            df[f'cvd_{window}'] = TradeFlowFeatures.cvd(df, window)
            df[f'cvd_norm_{window}'] = TradeFlowFeatures.cvd_normalized(df, window)
            df[f'delta_ma_ratio_{window}'] = TradeFlowFeatures.delta_ma_ratio(df, window)

        # Средний размер агрессивной сделки
        if 'num_trades' in df.columns:
            df['aggressive_trade_size'] = TradeFlowFeatures.aggressive_trade_size(df)

        return df

    def _generate_oi_features(
        self,
        df: pd.DataFrame,
        oi_df: Optional[pd.DataFrame] = None,
        funding_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Добавить признаки Open Interest и Funding Rate.
        
        IMPORTANT: OI features are calculated at NATIVE resolution (15m) BEFORE merging
        to prevent zero-variance features from forward-filled 1m data.
        
        Args:
            df: DF с OHLCV данными (должен быть отсортирован по timestamp)
            oi_df: DF с данными OI (timestamp, sumOpenInterest, sumOpenInterestValue)
            funding_df: DF с данными funding rate

        Returns:
            DataFrame с добавленными колонками OI и Funding
        """
        # Process OI if available
        if oi_df is not None and not oi_df.empty:
            oi = oi_df.copy()
            oi['timestamp'] = pd.to_datetime(oi['timestamp'])
            oi = oi.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate OI features at NATIVE resolution (15m)
            # This ensures actual changes are captured, not forward-filled zeros
            
            # OI percentage change at 15m resolution
            oi['oi_change_pct'] = oi['sumOpenInterest'].pct_change().fillna(0)
            
            # OI Z-score using rolling window (detect anomalies)
            oi_mean = oi['sumOpenInterest'].rolling(10, min_periods=3).mean()
            oi_std = oi['sumOpenInterest'].rolling(10, min_periods=3).std()
            oi['oi_zscore'] = ((oi['sumOpenInterest'] - oi_mean) / oi_std.replace(0, np.nan)).fillna(0)
            
            # OI acceleration (rate of change of change)
            oi['oi_acceleration'] = oi['oi_change_pct'].diff().fillna(0)
            
            # OI momentum (cumulative change over last 4 periods = 1 hour)
            oi['oi_momentum'] = oi['oi_change_pct'].rolling(4, min_periods=1).sum()
            
            # Merge pre-calculated features to OHLCV
            df = pd.merge_asof(
                df.sort_values('timestamp'),
                oi[['timestamp', 'oi_change_pct', 'oi_zscore', 'oi_acceleration', 'oi_momentum']],
                on='timestamp',
                direction='backward',
                tolerance=pd.Timedelta('30min')
            )
        
        # Initialize/fill missing columns with zeros (AFTER merge to avoid duplicates)
        for col in ['oi_change_pct', 'oi_zscore', 'oi_acceleration', 'oi_momentum']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0)
        
        # Process Funding Rate if available
        if funding_df is not None and not funding_df.empty:
            fr = funding_df.copy()
            fr['timestamp'] = pd.to_datetime(fr['timestamp'])
            
            df = pd.merge_asof(
                df.sort_values('timestamp'),
                fr[['timestamp', 'fundingRate']].rename(columns={'fundingRate': 'funding_rate'}),
                on='timestamp',
                direction='backward',
                tolerance=pd.Timedelta('8h')
            )
        
        # Initialize funding_rate if not present
        if 'funding_rate' not in df.columns:
            df['funding_rate'] = 0.0
        else:
            df['funding_rate'] = df['funding_rate'].fillna(0)

        return df

    def _generate_volume_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация Volume Anomaly признаков.

        Args:
            df: DataFrame с исходными данными

        Returns:
            DataFrame с добавленными Volume Anomaly признаками
        """
        # Z-score объема для разных окон
        for window in self.feature_config.volume_zscore_windows:
            df[f'vol_zscore_{window}'] = VolumeAnomalyFeatures.volume_zscore(df, window)

        # Сезонный относительный объем
        df['rvol_seasonal'] = VolumeAnomalyFeatures.rvol_seasonal(df)

        # Ускорение объема
        df['vol_acceleration'] = VolumeAnomalyFeatures.vol_acceleration(df)

        # Volume spike
        df['vol_spike_20'] = VolumeAnomalyFeatures.volume_spike(df, window=20, threshold=2.0)
        df['vol_spike_60'] = VolumeAnomalyFeatures.volume_spike(df, window=60, threshold=2.5)

        return df

    def _generate_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация Price Action признаков.

        Args:
            df: DataFrame с исходными данными

        Returns:
            DataFrame с добавленными Price Action признаками
        """
        # Логарифмические доходности
        df['log_return'] = PriceActionFeatures.log_returns(df)

        # Волатильность Паркинсона
        df['parkinson_vol_20'] = PriceActionFeatures.parkinson_vol(df, window=20)

        # ATR
        df['atr_14'] = PriceActionFeatures.atr(df, window=14)

        # Пробои
        df['breakout_upper_20'] = PriceActionFeatures.breakout_upper(df, window=20)
        df['breakout_lower_20'] = PriceActionFeatures.breakout_lower(df, window=20)

        # Фитили
        df['upper_wick_ratio'] = PriceActionFeatures.upper_wick_ratio(df)
        df['lower_wick_ratio'] = PriceActionFeatures.lower_wick_ratio(df)
        df['candle_body_ratio'] = PriceActionFeatures.candle_body_ratio(df)

        df['rsi_14'] = PriceActionFeatures.rsi(df, window=14)
        df['rsi_7'] = PriceActionFeatures.rsi(df, window=7)

        df['bb_width_20'] = PriceActionFeatures.bollinger_band_width(df, window=20)
        df['bb_pct_b_20'] = PriceActionFeatures.bollinger_pct_b(df, window=20)

        df['vwap_dev_60'] = PriceActionFeatures.vwap_deviation(df, window=60)

        df['roc_5'] = PriceActionFeatures.rate_of_change(df, window=5)
        df['roc_10'] = PriceActionFeatures.rate_of_change(df, window=10)
        df['roc_30'] = PriceActionFeatures.rate_of_change(df, window=30)

        df['trade_count_zscore_60'] = PriceActionFeatures.trade_count_zscore(df, window=60)

        return df

    def _generate_market_regime_features(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Генерация Market Regime признаков.

        Args:
            df: DataFrame с данными альткоина
            btc_df: DataFrame с данными BTC

        Returns:
            DataFrame с добавленными Market Regime признаками
        """
        # Подготавливаем BTC данные
        btc_df = btc_df.copy()
        btc_df = btc_df.sort_values('timestamp').reset_index(drop=True)

        # Рассчитываем доходности
        df['log_return'] = PriceActionFeatures.log_returns(df)
        btc_df['log_return'] = PriceActionFeatures.log_returns(btc_df)

        # Объединяем по времени для корректного расчета корреляции
        merged = pd.merge(
            df[['timestamp', 'log_return']],
            btc_df[['timestamp', 'log_return']].rename(columns={'log_return': 'btc_return'}),
            on='timestamp',
            how='left'
        )

        # Скользящая корреляция с BTC
        window = self.feature_config.btc_corr_window
        df['btc_corr_30'] = MarketRegimeFeatures.rolling_correlation(
            merged['log_return'],
            merged['btc_return'],
            window=window
        )

        # Падение корреляции
        df['btc_corr_drop'] = MarketRegimeFeatures.correlation_drop(df['btc_corr_30'])

        # Относительная сила к BTC
        df['relative_strength_30'] = MarketRegimeFeatures.relative_strength(
            merged['log_return'],
            merged['btc_return'],
            window=30
        )

        # ATR-нормализованный спред
        df['atr_norm_spread'] = MarketRegimeFeatures.atr_normalized_spread(
            PriceActionFeatures.atr(df, 14),
            df['close'],
            PriceActionFeatures.atr(btc_df, 14),
            btc_df['close']
        )

        return df

    def _apply_shift_for_preventing_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        КРИТИЧЕСКИ ВАЖНО: Сдвигает все признаки на 1 период назад.

        Это предотвращает Data Leakage - модель видит состояние
        рынка строго ДО начала пампа/дампа.

        Args:
            df: DataFrame со всеми признаками

        Returns:
            DataFrame со сдвинутыми признаками
        """
        # Колонки, которые НЕ нужно сдвигать
        no_shift_cols = {
            'timestamp', 'symbol', 'close_time',
            'open', 'high', 'low', 'close',  # Сырые OHLCV остаются для справки
            'volume', 'quote_volume', 'num_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume'
        }

        # Определяем колонки для сдвига
        shift_cols = [col for col in df.columns if col not in no_shift_cols]

        logger.info(f"Applying shift(1) to {len(shift_cols)} feature columns to prevent data leakage")

        # Применяем shift
        for col in shift_cols:
            df[f'{col}_lag1'] = df[col].shift(1)

        df = df.drop(columns=shift_cols)

        lag1_cols = [c for c in df.columns if c.endswith('_lag1')]
        df[lag1_cols] = df[lag1_cols].fillna(0)

        df = df.iloc[1:].reset_index(drop=True)

        return df

    def merge_oi_funding_features(
        self, 
        features_df: pd.DataFrame, 
        oi_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Добавить признаки Open Interest и Funding Rate.
        
        Args:
            features_df: DF с основными признаками
            oi_data: Словарь {symbol: DataFrame(oi)}
        """
        all_rows = []
        
        for symbol, group in features_df.groupby('symbol'):
            oi_df = oi_data.get(symbol)
            if oi_df is None or oi_df.empty:
                # Если данных нет, заполняем нулями (или drop)
                group['oi_change_1h'] = 0.0
                group['funding_rate'] = 0.0
                all_rows.append(group)
                continue
            
            # Синхронизация времени
            oi_df = oi_df.copy()
            oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'])
            
            # Сортируем для merge_asof (точное совпадение по времени невозможно)
            # Используем forward fill (берем последнее известное значение OI)
            merged = pd.merge_asof(
                group.sort_values('timestamp'),
                oi_df[['timestamp', 'sumOpenInterest', 'sumOpenInterestValue']],
                on='timestamp',
                direction='backward', # Берем предыдущее известное значение
                tolerance=pd.Timedelta('1h') # Допускаем отставание до часа
            )
            
            # Признаки OI
            merged['oi_change_pct'] = merged['sumOpenInterest'].pct_change()
            merged['oi_zscore_30'] = (merged['sumOpenInterest'] - merged['sumOpenInterest'].rolling(30).mean()) / merged['sumOpenInterest'].rolling(30).std()
            
            all_rows.append(merged)
            
        return pd.concat(all_rows, ignore_index=True)

    def generate_features_for_multiple_symbols(
        self,
        data: Dict[str, pd.DataFrame],
        oi_data: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Сгенерировать признаки для множества символов.

        Args:
            data: Словарь {symbol: DataFrame} с OHLCV данными

        Returns:
            Объединенный DataFrame со всеми признаками
        """
        btc_symbol = self.config.btc_symbol
        btc_df = data.get(btc_symbol)

        if btc_df is None:
            logger.warning(f"BTC data ({btc_symbol}) not found. Market regime features will be missing.")
            logger.warning(f"Available symbols: {list(data.keys())}")

        all_features = []

        for symbol, df in data.items():
            if symbol == btc_symbol:
                continue  # Пропускаем BTC в основном датасете

            logger.info(f"Generating features for {symbol}")
            sym_oi_data = (oi_data or {}).get(symbol, {})
            features_df = self.generate_all_features(df, btc_df)
            all_features.append(features_df)

        # Объединяем все символы
        result = pd.concat(all_features, ignore_index=True)

        logger.info(f"Generated features for {len(all_features)} symbols, total rows: {len(result)}")

        return result

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Получить словарь групп признаков для анализа важности.

        Returns:
            Словарь {group_name: [feature_names]}
        """
        return self.feature_names


def prepare_training_data(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Подготовить данные для обучения: объединить признаки с таргетом.

    Args:
        features_df: DataFrame со всеми признаками
        labels_df: DataFrame с метками (timestamp, symbol, label)

    Returns:
        Кортеж (X, y) для обучения
    """
    # Унифицируем форматы timestamp
    features_df = features_df.copy()
    labels_df = labels_df.copy()

    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])

    # Объединяем по timestamp и symbol
    merged = pd.merge(
        features_df,
        labels_df,
        on=['timestamp', 'symbol'],
        how='inner'  # Только наблюдения с метками
    )

    logger.info(f"Merged dataset size: {len(merged)} rows")

    if len(merged) == 0:
        # Пробуем альтернативный подход - округляем timestamp до минут
        logger.warning("No exact matches, trying rounded timestamp merge...")

        features_df['ts_rounded'] = features_df['timestamp'].dt.floor('min')
        labels_df['ts_rounded'] = labels_df['timestamp'].dt.floor('min')

        merged = pd.merge(
            features_df,
            labels_df,
            on=['ts_rounded', 'symbol'],
            how='inner',
            suffixes=('', '_label')
        )

        # Используем label из labels_df
        if 'label_label' in merged.columns:
            merged['label'] = merged['label_label']

        merged = merged.drop(columns=['ts_rounded'], errors='ignore')
        logger.info(f"Rounded merge dataset size: {len(merged)} rows")

    if len(merged) == 0:
        logger.error("Still no matches! Check timestamps in labels and features.")
        return pd.DataFrame(), pd.Series()

    # Отделяем признаки от таргета
    # Исключаем не-feature колонки
    exclude_cols = {
        'timestamp', 'symbol', 'label', 'close_time', 'label_label', 'ts_rounded',
        # Сырые OHLCV колонки - не используем как features!
        'open', 'high', 'low', 'close', 'volume', 'quote_volume',
        'num_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume'
    }
    feature_cols = [col for col in merged.columns if col not in exclude_cols]

    X = merged[feature_cols].copy()
    y = merged['label'].copy()

    # Заполняем NaN
    X = X.fillna(0)

    # Убираем колонки с нулевой дисперсией
    zero_var_cols = X.columns[X.std() == 0]
    if len(zero_var_cols) > 0:
        logger.warning(f"Dropping zero variance columns: {list(zero_var_cols)}")
        X = X.drop(columns=zero_var_cols)

    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    return X, y
