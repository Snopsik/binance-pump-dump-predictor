"""
Динамический сканер Binance Futures.
Никаких хардкодов — только твои критерии + реальные данные.
"""

import asyncio
import pandas as pd
import numpy as np
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ТВОИ КРИТЕРИИ — меняй под себя
# =============================================================================

@dataclass
class ScanCriteria:
    """
    Критерии отбора токенов. Всё настраивается.
    """
    # Минимальная дневная волатильность (High/Low диапазон)
    min_volatility_pct: float = 5.0       # % — убирает стейблы и скучные токены

    # Минимальный суточный объём (USDT)
    min_daily_volume_usdt: float = 1_000_000   # $1M — убирает неликвид

    # Порог для детекции памп/дамп (% за lookahead минут)
    pump_dump_threshold_pct: float = 5.0  # 5% за 15 минут = памп

    # Минимальное кол-во пампов ИЛИ дампов за период
    min_pump_dump_events: int = 3         # хотя бы 3 движения за период

    # Lookahead для детекции (в свечах = минутах)
    lookahead_minutes: int = 15

    # Минимальное кол-во свечей (меньше = недостаточно данных)
    min_candles: int = 500

    # Фильтр по максимальному объёму (убирает BTC/ETH)
    max_daily_volume_usdt: float = 5_000_000_000  # $5B

    # Минимальный manipulation score (0.0–1.0)
    min_manipulation_score: float = 0.0   # 0 = без фильтра


# Дефолтные критерии (меняй как хочешь)
DEFAULT_CRITERIA = ScanCriteria()

# Агрессивные критерии (только реальные шитки с MM)
AGGRESSIVE_CRITERIA = ScanCriteria(
    min_volatility_pct=8.0,
    min_daily_volume_usdt=500_000,
    pump_dump_threshold_pct=8.0,
    min_pump_dump_events=5,
    min_manipulation_score=0.3,
)

# Мягкие критерии (больше токенов)
SOFT_CRITERIA = ScanCriteria(
    min_volatility_pct=3.0,
    min_daily_volume_usdt=200_000,
    pump_dump_threshold_pct=3.0,
    min_pump_dump_events=1,
)

# Символы которые НИКОГДА не нужны (стейблы, мажоры)
EXCLUDED_SYMBOLS: set = {
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "USDT/USDT", "USDC/USDT", "BUSD/USDT", "DAI/USDT", "TUSD/USDT",
    "BTCB/USDT", "WBTC/USDT", "STETH/USDT",
}


# =============================================================================
# Результат анализа одного токена
# =============================================================================

@dataclass
class TokenProfile:
    symbol: str
    volatility_pct: float        # Средняя дневная волатильность %
    daily_volume_usdt: float     # Средний дневной объём USDT
    pump_count: int              # Кол-во пампов за период
    dump_count: int              # Кол-во дампов за период
    pump_dump_threshold: float   # Порог который использовался
    volume_spikes: int           # Кол-во аномальных всплесков объёма
    taker_imbalance: float       # Avg taker imbalance (0.5 = нейтрально)
    manipulation_score: float    # Итоговый score 0–1
    candles_count: int           # Сколько свечей в данных

    @property
    def total_events(self) -> int:
        return self.pump_count + self.dump_count

    def __str__(self) -> str:
        return (
            f"{self.symbol:<18} vol={self.volatility_pct:>6.1f}%  "
            f"vol$={self.daily_volume_usdt/1e6:>6.1f}M  "
            f"pumps={self.pump_count:>4}  dumps={self.dump_count:>4}  "
            f"score={self.manipulation_score:.2f}"
        )


# =============================================================================
# Получение всех символов Binance Futures
# =============================================================================

async def get_all_futures_symbols(collector) -> List[str]:
    """Все активные PERPETUAL USDT-M фьючерсы с Binance."""
    try:
        info = await collector._make_request("/fapi/v1/exchangeInfo", {})
        symbols = []
        for s in info.get("symbols", []):
            if s["status"] == "TRADING" and s["contractType"] == "PERPETUAL":
                if s["quoteAsset"] == "USDT":
                    sym = f"{s['baseAsset']}/USDT"
                    if sym not in EXCLUDED_SYMBOLS:
                        symbols.append(sym)
        logger.info(f"Binance Futures: {len(symbols)} active symbols")
        return symbols
    except Exception as e:
        logger.error(f"Failed to get symbols: {e}")
        return []


# =============================================================================
# Анализ одного токена
# =============================================================================
def setup_windows_dns() -> None:
    """
    Настроить DNS резолвер для Windows.

    На Windows иногда возникают проблемы с DNS резолвингом.
    Эта функция пытается их решить.
    """
    if platform.system() == 'Windows':
        # Устанавливаем event loop policy для Windows
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            logger.info("Windows ProactorEventLoop policy set")
            
def analyze_token(
    symbol: str,
    df: pd.DataFrame,
    criteria: ScanCriteria,
) -> Optional[TokenProfile]:
    """Анализировать токен по критериям. None = не прошёл фильтры."""
    try:
        if len(df) < criteria.min_candles:
            return None

        df = df.copy().sort_values("timestamp").reset_index(drop=True)

        # ← ФИКС: гарантируем datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Волатильность (Parkinson H/L)
        df["hl_range"] = (df["high"] - df["low"]) / df["low"].replace(0, np.nan)
        date_groups = df.groupby(df["timestamp"].dt.date)
        avg_vol_pct = float(date_groups["hl_range"].mean().mean() * 100)

        logger.debug(f"{symbol}: vol={avg_vol_pct:.2f}% (need >={criteria.min_volatility_pct}%)")

        if avg_vol_pct < criteria.min_volatility_pct:
            return None

        # Объём
        vol_col = "quote_volume" if "quote_volume" in df.columns else "volume"
        avg_daily_vol = float(date_groups[vol_col].sum().mean())

        logger.debug(f"{symbol}: vol$={avg_daily_vol/1e6:.2f}M (need >={criteria.min_daily_volume_usdt/1e6:.2f}M)")

        if avg_daily_vol < criteria.min_daily_volume_usdt:
            return None
        if avg_daily_vol > criteria.max_daily_volume_usdt:
            return None

        # Пампы / дампы
        threshold = criteria.pump_dump_threshold_pct / 100.0
        la = criteria.lookahead_minutes
        df["future_ret"] = df["close"].pct_change(la).shift(-la)
        pumps = int((df["future_ret"] > threshold).sum())
        dumps  = int((df["future_ret"] < -threshold).sum())

        logger.debug(f"{symbol}: pumps={pumps} dumps={dumps} (need >={criteria.min_pump_dump_events})")

        if (pumps + dumps) < criteria.min_pump_dump_events:
            return None

        # Volume spike z-score
        roll_mean = df["volume"].rolling(60, min_periods=10).mean()
        roll_std  = df["volume"].rolling(60, min_periods=10).std().replace(0, np.nan)
        df["vol_z"] = (df["volume"] - roll_mean) / roll_std
        vol_spikes = int((df["vol_z"] > 3).sum())

        # Taker flow imbalance
        if "taker_buy_base_volume" in df.columns:
            taker_r = df["taker_buy_base_volume"] / df["volume"].replace(0, np.nan)
            avg_imbalance = float((taker_r - 0.5).abs().mean())
        else:
            avg_imbalance = 0.0

        # Manipulation score (0-1)
        score = (
            min(avg_vol_pct / 50.0, 1.0)         * 0.25
            + min((pumps + dumps) / 100.0, 1.0)  * 0.35
            + min(vol_spikes / 200.0, 1.0)       * 0.20
            + min(avg_imbalance * 4.0, 1.0)      * 0.20
        )

        if score < criteria.min_manipulation_score:
            return None

        return TokenProfile(
            symbol=symbol,
            volatility_pct=avg_vol_pct,
            daily_volume_usdt=avg_daily_vol,
            pump_count=pumps,
            dump_count=dumps,
            pump_dump_threshold=threshold,
            volume_spikes=vol_spikes,
            taker_imbalance=avg_imbalance,
            manipulation_score=score,
            candles_count=len(df),
        )

    except Exception as e:
        # ← ФИКС: WARNING вместо DEBUG чтобы видеть реальные ошибки
        logger.warning(f"analyze_token({symbol}) error: {e}")
        return None


# =============================================================================
# Главная функция сканирования
# =============================================================================

async def scan_all_futures(
    collector,
    criteria: ScanCriteria = DEFAULT_CRITERIA,
    days: int = 7,
    top_n: int = 50,
    sort_by: str = "volatility",   # "volatility" | "score" | "events" | "volume"
) -> List[TokenProfile]:
    """
    Сканировать ВСЕ Binance Futures и вернуть топ-N по твоим критериям.

    Args:
        collector:  BinanceDataCollector (активный)
        criteria:   Критерии фильтрации (ScanCriteria)
        days:       Дней истории
        top_n:      Сколько токенов вернуть
        sort_by:    По чему сортировать результат

    Returns:
        Список TokenProfile, отсортированный по sort_by
    """
    all_symbols = await get_all_futures_symbols(collector)
    logger.info(f"Scanning {len(all_symbols)} symbols with criteria: {criteria}")

    profiles: List[TokenProfile] = []

    for i, symbol in enumerate(all_symbols, 1):
        try:
            # Используем максимальный лимит 1500 свечей (25 часов)
            # Этого достаточно для анализа волатильности и pump/dump событий
            df = await collector.fetch_ohlcv(symbol, limit=1500)
            if df.empty:
                continue

            profile = analyze_token(symbol, df, criteria)
            if profile is not None:
                profiles.append(profile)
                logger.info(f"[{i}/{len(all_symbols)}] ✅ {profile}")
            else:
                logger.debug(f"[{i}/{len(all_symbols)}] ❌ {symbol} filtered out")

        except Exception as e:
            logger.debug(f"Error on {symbol}: {e}")
            continue

    # Сортировка
    sort_keys = {
        "volatility": lambda x: x.volatility_pct,
        "score":      lambda x: x.manipulation_score,
        "events":     lambda x: x.total_events,
        "volume":     lambda x: x.daily_volume_usdt,
    }
    key_fn = sort_keys.get(sort_by, sort_keys["volatility"])
    profiles.sort(key=key_fn, reverse=True)

    logger.info(f"Scan complete: {len(profiles)} tokens passed filters")
    return profiles[:top_n]


# =============================================================================
# Хелпер: динамический порог для конкретного токена
# =============================================================================

def get_dynamic_threshold(profile: TokenProfile) -> float:
    """
    Автоматически рассчитать порог памп/дамп на основе реальной волатильности.
    Не нужно захардкоживать — берём из данных.
    """
    vol = profile.volatility_pct
    if vol > 30:
        return 0.15
    elif vol > 20:
        return 0.10
    elif vol > 10:
        return 0.08
    elif vol > 5:
        return 0.06
    else:
        return 0.04


def get_shitcoin_threshold(symbol: str) -> float:
    """
    Дефолт 8% — используется только если нет TokenProfile.
    В идеале использовать get_dynamic_threshold(profile).
    """
    return 0.08


# =============================================================================
# Live Scanner
# =============================================================================

class ShitcoinLiveScanner:
    """Live мониторинг токенов из результатов скана."""

    def __init__(
        self,
        profiles: List[TokenProfile],
    ):
        # Используем динамические пороги из реальных данных
        self.thresholds = {
            p.symbol: get_dynamic_threshold(p)
            for p in profiles
        }
        self._buffer: Dict[str, pd.DataFrame] = {}

    def update(self, symbol: str, ohlcv: pd.DataFrame) -> Optional[Dict]:
        """Обновить буфер и проверить на аномалии."""
        if symbol not in self._buffer:
            self._buffer[symbol] = ohlcv.copy()
            return None

        self._buffer[symbol] = (
            pd.concat([self._buffer[symbol], ohlcv])
            .drop_duplicates(subset=["timestamp"])
            .tail(500)
        )

        df = self._buffer[symbol]
        if len(df) < 60:
            return None

        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        threshold = self.thresholds.get(symbol, 0.08)

        price_change = (latest["close"] - prev["close"]) / prev["close"]
        vol_ma = df["volume"].rolling(20).mean().iloc[-1]
        vol_spike = latest["volume"] / vol_ma if vol_ma > 0 else 1.0
        taker_r = (
            latest["taker_buy_base_volume"] / latest["volume"]
            if latest.get("volume", 0) > 0 else 0.5
        )

        result = {}
        if price_change > threshold:
            result["pump"] = {
                "price_change": price_change,
                "volume_spike": vol_spike,
                "taker_ratio":  taker_r,
                "timestamp":    latest["timestamp"],
                "price":        latest["close"],
            }
        if price_change < -threshold:
            result["dump"] = {
                "price_change": price_change,
                "volume_spike": vol_spike,
                "taker_ratio":  taker_r,
                "timestamp":    latest["timestamp"],
                "price":        latest["close"],
            }
        if vol_spike > 5 and abs(price_change) < 0.02:
            result["volume_warning"] = {
                "volume_spike": vol_spike,
                "taker_ratio":  taker_r,
                "timestamp":    latest["timestamp"],
            }

        return result if result else None


# =============================================================================
# generate_shitcoin_labels — для dual-train
# =============================================================================

def generate_shitcoin_labels(
    data: Dict[str, pd.DataFrame],
    profiles: Optional[List[TokenProfile]] = None,
    default_threshold_pct: float = 5.0,
    lookahead: int = 15,
) -> pd.DataFrame:
    """
    Генерация меток с динамическими порогами из реального скана.

    Если profiles переданы — использует порог из данных каждого токена.
    Иначе использует default_threshold_pct для всех.
    """
    # Строим словарь порогов
    thresholds = {}
    if profiles:
        thresholds = {p.symbol: get_dynamic_threshold(p) for p in profiles}

    all_labels = []

    for symbol, df in data.items():
        if symbol in EXCLUDED_SYMBOLS:
            continue

        threshold = thresholds.get(symbol, default_threshold_pct / 100.0)

        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        df["future_return"] = df["close"].pct_change(lookahead).shift(-lookahead)

        labels = pd.DataFrame({
            "timestamp": df["timestamp"],
            "symbol":    symbol,
            "label":     0,
        })
        labels.loc[df["future_return"] > threshold,  "label"] = 1
        labels.loc[df["future_return"] < -threshold, "label"] = -1
        labels = labels.iloc[:-lookahead].dropna()

        all_labels.append(labels)

        n_p = int((labels["label"] == 1).sum())
        n_d = int((labels["label"] == -1).sum())
        n   = len(labels)
        if n > 0:
            logger.info(
                f"  {symbol}: Pumps={n_p} ({100*n_p/n:.1f}%)  "
                f"Dumps={n_d} ({100*n_d/n:.1f}%)  "
                f"Threshold=±{threshold*100:.1f}%"
            )

    if not all_labels:
        return pd.DataFrame(columns=["timestamp", "symbol", "label"])

    result = pd.concat(all_labels, ignore_index=True)
    n_p = int((result["label"] == 1).sum())
    n_d = int((result["label"] == -1).sum())
    n_n = int((result["label"] == 0).sum())
    n   = len(result)
    print(f"\nLabels: total={n}  pumps={n_p} ({100*n_p/n:.1f}%)  "
          f"dumps={n_d} ({100*n_d/n:.1f}%)  neutral={n_n} ({100*n_n/n:.1f}%)")
    return result


if __name__ == "__main__":
    print("ScanCriteria presets:")
    print(f"  DEFAULT:    vol≥{DEFAULT_CRITERIA.min_volatility_pct}%  "
          f"vol$≥{DEFAULT_CRITERIA.min_daily_volume_usdt/1e6:.1f}M  "
          f"threshold={DEFAULT_CRITERIA.pump_dump_threshold_pct}%")
    print(f"  AGGRESSIVE: vol≥{AGGRESSIVE_CRITERIA.min_volatility_pct}%  "
          f"vol$≥{AGGRESSIVE_CRITERIA.min_daily_volume_usdt/1e6:.1f}M  "
          f"threshold={AGGRESSIVE_CRITERIA.pump_dump_threshold_pct}%")
    print(f"  SOFT:       vol≥{SOFT_CRITERIA.min_volatility_pct}%  "
          f"vol$≥{SOFT_CRITERIA.min_daily_volume_usdt/1e6:.1f}M  "
          f"threshold={SOFT_CRITERIA.pump_dump_threshold_pct}%")