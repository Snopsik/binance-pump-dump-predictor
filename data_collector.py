"""
Асинхронный сборщик данных с Binance Futures API с поддержкой Windows.

Исправлены проблемы:
- DNS resolution на Windows
- SSL сертификаты
- Прокси поддержка

OPTIMIZED: Uses orjson for faster JSON parsing (~2-3x faster than standard json).
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
import time
import ssl
import socket
import platform
from dataclasses import dataclass
import warnings

# Try to use orjson for faster JSON parsing (optional but recommended)
try:
    import orjson as json
    ORJSON_AVAILABLE = True
except ImportError:
    import json
    ORJSON_AVAILABLE = False

from config import PipelineConfig, TimeFrame

# Настройка логирования
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# =============================================================================
# Windows DNS/SSL Fixes
# =============================================================================

def create_ssl_context() -> ssl.SSLContext:
    """
    Создать SSL контекст с правильными настройками для Windows.

    Решает проблемы с SSL сертификатами на Windows.
    """
    ssl_context = ssl.create_default_context()

    # На Windows могут быть проблемы с сертификатами
    if platform.system() == 'Windows':
        # Пытаемся найти сертификаты
        try:
            import certifi
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            logger.info("Using certifi certificates")
        except ImportError:
            logger.warning("certifi not installed, using default SSL context")

    return ssl_context


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

        # Альтернативно - используем SelectorEventLoop
        # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def test_binance_connection() -> bool:
    """
    Проверить подключение к Binance API.

    Returns:
        True если подключение работает
    """
    ssl_context = create_ssl_context()

    connector = aiohttp.TCPConnector(
        ssl=ssl_context,
        family=socket.AF_INET,  # IPv4 только
        force_close=True,
        enable_cleanup_closed=True
    )

    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            async with session.get('https://fapi.binance.com/fapi/v1/ping') as resp:
                if resp.status == 200:
                    logger.info("✓ Binance API connection successful!")
                    return True
                else:
                    logger.error(f"✗ Binance API returned status {resp.status}")
                    return False
    except aiohttp.ClientError as e:
        logger.error(f"✗ Binance API connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False


# =============================================================================
# Константы и структура данных OHLCV
# =============================================================================

@dataclass
class BinanceKlineColumns:
    """Индексы полей в ответе Binance klines API."""
    OPEN_TIME: int = 0
    OPEN: int = 1
    HIGH: int = 2
    LOW: int = 3
    CLOSE: int = 4
    VOLUME: int = 5
    CLOSE_TIME: int = 6
    QUOTE_VOLUME: int = 7
    NUM_TRADES: int = 8
    TAKER_BUY_BASE_VOLUME: int = 9
    TAKER_BUY_QUOTE_VOLUME: int = 10


KLINE_COLS = BinanceKlineColumns()

OHLCV_COLUMN_NAMES = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'num_trades',
    'taker_buy_base_volume', 'taker_buy_quote_volume',
]


class RateLimiter:
    """Асинхронный rate limiter для соблюдения лимитов API Binance."""

    def __init__(self, requests_per_second: int = 50):
        self.requests_per_second = requests_per_second
        self._tokens = float(requests_per_second)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        async with self._lock:
            while True:
                now = time.time()
                elapsed = now - self._last_update
                self._tokens = min(
                    self.requests_per_second,
                    self._tokens + elapsed * self.requests_per_second
                )
                self._last_update = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                sleep_time = (tokens - self._tokens) / self.requests_per_second
                await asyncio.sleep(sleep_time)


class BinanceDataCollector:
    """
    Асинхронный сборщик данных с Binance Futures.

    Исправлен для работы на Windows с проблемами DNS/SSL.
    """

    TIMEFRAME_MS = {
        TimeFrame.MINUTE_1: 60_000,
        TimeFrame.MINUTE_5: 300_000,
        TimeFrame.MINUTE_15: 900_000,
        TimeFrame.HOUR_1: 3_600_000,
    }

    MAX_KLINES_PER_REQUEST = 1500

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.base_url = config.binance.base_url
        self.rate_limiter = RateLimiter(config.binance.rate_limit_requests)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache_dir = Path(config.data_cache_dir)

        # SSL контекст для Windows
        self._ssl_context = create_ssl_context()

    async def __aenter__(self) -> 'BinanceDataCollector':
        """Вход в контекстный менеджер с правильными настройками для Windows."""

        # Настраиваем Windows DNS
        setup_windows_dns()

        # Создаем connector с правильными настройками
        connector = aiohttp.TCPConnector(
            ssl=self._ssl_context,
            family=socket.AF_INET,  # IPv4 только - решает DNS проблемы
            force_close=True,
            enable_cleanup_closed=True,
            limit=10,  # Лимит соединений
            limit_per_host=5
        )

        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30, connect=10),
            headers={
                'X-MBX-APIKEY': self.config.binance.api_key or '',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session:
            await self._session.close()

    def _symbol_to_binance(self, symbol: str) -> str:
        return symbol.replace('/', '').upper()

    def _get_cache_path(self, symbol: str, start_time: int, end_time: int) -> Path:
        safe_symbol = symbol.replace('/', '_')
        return self._cache_dir / f"{safe_symbol}_{start_time}_{end_time}.parquet"

    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        retries: int = 3
    ) -> Any:
        """
        Выполнить HTTP запрос к API Binance с retry логикой.

        Добавлена обработка ошибок DNS и SSL.
        Uses orjson for faster JSON parsing when available.
        """
        await self.rate_limiter.acquire()

        url = f"{self.base_url}{endpoint}"

        for attempt in range(retries):
            try:
                async with self._session.get(url, params=params) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limit exceeded, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status == 418:
                        logger.error("IP temporarily banned by Binance")
                        raise aiohttp.ClientError("IP banned")

                    response.raise_for_status()
                    
                    # Use orjson for faster parsing if available
                    if ORJSON_AVAILABLE:
                        raw_data = await response.read()
                        return json.loads(raw_data)
                    else:
                        return await response.json()

            except aiohttp.ClientConnectorError as e:
                logger.error(f"Connection error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

            except aiohttp.ClientSSLError as e:
                logger.error(f"SSL error: {e}")
                # Пробуем без SSL верификации как fallback
                if attempt == retries - 1:
                    logger.warning("Trying without SSL verification...")
                    connector = aiohttp.TCPConnector(ssl=False)
                    async with aiohttp.ClientSession(connector=connector) as session:
                        async with session.get(url, params=params) as response:
                            if ORJSON_AVAILABLE:
                                raw_data = await response.read()
                                return json.loads(raw_data)
                            else:
                                return await response.json()

            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def fetch_ohlcv(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1500
    ) -> pd.DataFrame:
        """Загрузить OHLCV данные с taker flow метриками."""
        binance_symbol = self._symbol_to_binance(symbol)
        interval = self.config.timeframe.value

        params = {
            'symbol': binance_symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        logger.info(f"Fetching OHLCV for {symbol} ({binance_symbol})")

        data = await self._make_request('/fapi/v1/klines', params)

        if not data:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.iloc[:, :11]
        df.columns = OHLCV_COLUMN_NAMES

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume',
                    'quote_volume', 'taker_buy_base_volume',
                    'taker_buy_quote_volume']:
            df[col] = df[col].astype(float)

        df['num_trades'] = df['num_trades'].astype(int)
        df['symbol'] = symbol

        logger.info(f"Fetched {len(df)} candles for {symbol}")

        return df

    async def fetch_ohlcv_range(
        self,
        symbol: str,
        days: int = 90
    ) -> pd.DataFrame:
        """Загрузить OHLCV данные за указанный период."""
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        cache_path = self._get_cache_path(symbol, start_time, end_time)
        if cache_path.exists():
            logger.info(f"Loading {symbol} from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        timeframe_ms = self.TIMEFRAME_MS[self.config.timeframe]
        all_data = []

        current_start = start_time

        while current_start < end_time:
            chunk_end = min(
                current_start + self.MAX_KLINES_PER_REQUEST * timeframe_ms,
                end_time
            )

            try:
                chunk = await self.fetch_ohlcv(
                    symbol,
                    start_time=current_start,
                    end_time=chunk_end,
                    limit=self.MAX_KLINES_PER_REQUEST
                )

                if chunk.empty:
                    break

                all_data.append(chunk)
                current_start = int(chunk['timestamp'].iloc[-1].timestamp() * 1000) + timeframe_ms
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching chunk for {symbol}: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['timestamp', 'symbol'])
        result = result.sort_values('timestamp').reset_index(drop=True)

        result.to_parquet(cache_path, index=False)
        logger.info(f"Cached {len(result)} candles for {symbol}")

        return result

    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        days: int = 90,
        include_btc: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Последовательная загрузка с задержками."""
        unique_symbols = set(symbols)
        if include_btc:
            unique_symbols.add(self.config.btc_symbol)

        logger.info(f"Fetching data for {len(unique_symbols)} symbols")

        data = {}
        total = len(unique_symbols)

        # ПОСЛЕДОВАТЕЛЬНАЯ загрузка вместо параллельной
        for i, symbol in enumerate(unique_symbols, 1):
            try:
                logger.info(f"Fetching [{i}/{total}] {symbol}...")
                result = await self.fetch_ohlcv_range(symbol, days)

                if not result.empty:
                    data[symbol] = result
                    logger.info(f"  OK: {len(result)} candles")

                # КЛЮЧЕВОЕ: Задержка 1 секунда между запросами!
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"  ERROR: {symbol}: {e}")
                continue

        return data

    async def get_latest_candle(self, symbol: str) -> Optional[pd.DataFrame]:
        """Получить последнюю закрытую свечу."""
        try:
            df = await self.fetch_ohlcv(symbol, limit=2)
            if len(df) >= 2:
                return df.iloc[[-2]]
            return None
        except Exception as e:
            logger.error(f"Error fetching latest candle for {symbol}: {e}")
            return None


async def collect_training_data(
    config: PipelineConfig,
    target_symbols: List[str]
) -> Dict[str, pd.DataFrame]:
    """Главная функция для сбора данных обучения."""

    # Тестируем подключение сначала
    logger.info("Testing Binance API connection...")
    if not await test_binance_connection():
        logger.error("""
╔═══════════════════════════════════════════════════════════════════════╗
║                    CONNECTION FAILED                                  ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Возможные решения для Windows:                                      ║
║                                                                       ║
║  1. Проверьте интернет соединение                                    ║
║                                                                       ║
║  2. Установите сертификаты:                                          ║
║     pip install certifi                                              ║
║                                                                       ║
║  3. Используйте VPN если Binance заблокирован в вашем регионе       ║
║                                                                       ║
║  4. Проверьте фаервол/антивирус                                      ║
║                                                                       ║
║  5. Попробуйте альтернативный DNS:                                   ║
║     - Google: 8.8.8.8, 8.8.4.4                                       ║
║     - Cloudflare: 1.1.1.1, 1.0.0.1                                   ║
║                                                                       ║
║  6. Используйте демо-режим:                                          ║
║     python main.py demo                                              ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
        """)
        return {}

    async with BinanceDataCollector(config) as collector:
        data = await collector.fetch_multiple_symbols(
            symbols=target_symbols,
            days=config.history_days,
            include_btc=True
        )
    return data


def load_labels(labels_path: str) -> pd.DataFrame:
    """Загрузить файл с метками пампов/дампов."""
    df = pd.read_csv(labels_path)

    required_cols = ['timestamp', 'symbol', 'label']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labels file: {missing}")

    if df['timestamp'].dtype in ['int64', 'float64']:
        if df['timestamp'].max() > 1e12:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    valid_labels = {1, -1, 0}
    invalid_labels = set(df['label'].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")

    return df


def get_unique_symbols_from_labels(labels_df: pd.DataFrame) -> List[str]:
    """Извлечь уникальные символы из файла меток."""
    return labels_df['symbol'].unique().tolist()
