"""
Сборщик Open Interest + Funding Rate с Binance Futures.
Binance API Documentation:
- Open Interest: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest
- Funding Rate: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History
"""
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
import time

logger = logging.getLogger(__name__)

FAPI_BASE = "https://fapi.binance.com"

# Binance API limits
MAX_OI_LIMIT = 500  # Maximum for openInterestHist endpoint
MAX_FUNDING_LIMIT = 1000  # Maximum for fundingRate endpoint


async def fetch_current_open_interest(
    session: aiohttp.ClientSession,
    symbol: str,  # формат: SIRENUSDT
) -> Optional[Dict]:
    """
    Получить текущий Open Interest для символа.
    Endpoint: GET /fapi/v1/openInterest
    
    Args:
        session: aiohttp session
        symbol: Trading pair symbol (e.g., "SIRENUSDT")
    
    Returns:
        Dict with openInterest or None on error
    """
    url = f"{FAPI_BASE}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                text = await r.text()
                logger.debug(f"Current OI failed for {symbol}: status={r.status}, response={text}")
                return None
            data = await r.json()
            
        if "openInterest" in data:
            return {
                "symbol": symbol,
                "openInterest": float(data["openInterest"]),
                "timestamp": datetime.utcnow()
            }
        return None
        
    except Exception as e:
        logger.debug(f"Current OI fetch error for {symbol}: {e}")
        return None


async def fetch_oi_history(
    session: aiohttp.ClientSession,
    symbol: str,  # формат: SIRENUSDT
    period: str = "15m",
    limit: int = 500,  # Max 500 for this endpoint
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """
    Исторический Open Interest с /futures/data/openInterestHist.
    
    IMPORTANT: Binance limits limit to 500 maximum!
    
    Args:
        session: aiohttp session
        symbol: Trading pair symbol (e.g., "SIRENUSDT")
        period: Time interval - "5m","15m","30m","1h","2h","4h","6h","12h","1d"
        limit: Number of records (max 500)
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
    
    Returns:
        DataFrame with columns: timestamp, sumOpenInterest, sumOpenInterestValue
    """
    # Ensure limit doesn't exceed max
    limit = min(limit, MAX_OI_LIMIT)
    
    url = f"{FAPI_BASE}/futures/data/openInterestHist"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }
    
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                text = await r.text()
                logger.debug(f"OI history failed for {symbol}: status={r.status}, response={text[:200]}")
                return pd.DataFrame()
            data = await r.json()
        
        if not data or (isinstance(data, dict) and "code" in data):
            logger.debug(f"OI history empty for {symbol}: {data}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        if df.empty:
            return df
            
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
        
        return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].sort_values("timestamp")
        
    except Exception as e:
        logger.debug(f"OI history fetch error for {symbol}: {e}")
        return pd.DataFrame()


async def fetch_oi_history_extended(
    session: aiohttp.ClientSession,
    symbol: str,
    period: str = "15m",
    days: int = 30,
) -> pd.DataFrame:
    """
    Получить расширенную историю Open Interest за несколько дней.
    Делает несколько запросов чтобы обойти лимит в 500 записей.
    
    Args:
        session: aiohttp session
        symbol: Trading pair symbol
        period: Time interval
        days: Number of days to fetch
    
    Returns:
        DataFrame with historical OI data
    """
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    # Calculate period in milliseconds
    period_ms = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }.get(period, 15 * 60 * 1000)
    
    # Calculate how many requests we need
    total_records = (end_time - start_time) / period_ms
    requests_needed = max(1, int(total_records / MAX_OI_LIMIT) + 1)
    
    logger.info(f"Fetching OI history for {symbol}: {days}d, period={period}, ~{requests_needed} requests")
    
    for i in range(min(requests_needed, 10)):  # Max 10 requests to avoid rate limits
        chunk_end = min(current_start + MAX_OI_LIMIT * period_ms, end_time)
        
        df = await fetch_oi_history(
            session, symbol, period,
            limit=MAX_OI_LIMIT,
            start_time=current_start,
            end_time=chunk_end
        )
        
        if df.empty:
            break
            
        all_data.append(df)
        current_start = int(df["timestamp"].iloc[-1].timestamp() * 1000) + period_ms
        
        if current_start >= end_time:
            break
            
        # Rate limiting
        await asyncio.sleep(0.2)
    
    if not all_data:
        return pd.DataFrame()
        
    result = pd.concat(all_data, ignore_index=True)
    result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    
    return result


async def fetch_funding_rate(
    session: aiohttp.ClientSession,
    symbol: str,
    limit: int = 1000,  # Max 1000
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """
    История funding rate с /fapi/v1/fundingRate.
    
    Args:
        session: aiohttp session
        symbol: Trading pair symbol (e.g., "SIRENUSDT")
        limit: Number of records (max 1000)
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
    
    Returns:
        DataFrame with columns: timestamp, fundingRate
    """
    limit = min(limit, MAX_FUNDING_LIMIT)
    
    url = f"{FAPI_BASE}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                text = await r.text()
                logger.debug(f"Funding rate failed for {symbol}: status={r.status}, response={text[:200]}")
                return pd.DataFrame()
            data = await r.json()
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        if df.empty:
            return df
            
        df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
        df["fundingRate"] = df["fundingRate"].astype(float)
        
        return df[["timestamp", "fundingRate"]].sort_values("timestamp")
        
    except Exception as e:
        logger.debug(f"Funding rate fetch error for {symbol}: {e}")
        return pd.DataFrame()


async def collect_oi_funding_batch(
    symbols: list,  # ["SIRENUSDT", "TAKEUSDT", ...]
    period: str = "15m",
    days: int = 30,
) -> Dict[str, Dict]:
    """
    Параллельно собираем OI + Funding для списка символов.
    
    Args:
        symbols: List of trading pair symbols in Binance format (e.g., ["SIRENUSDT"])
        period: Time interval for OI history
        days: Number of days to fetch
    
    Returns:
        {"SIREN/USDT": {"oi": df, "funding": df}, ...}
    """
    logger.info(f"Collecting OI/Funding for {len(symbols)} symbols, period={period}, days={days}")
    
    async with aiohttp.ClientSession() as session:
        # Fetch OI history (extended to cover multiple days)
        tasks_oi = [fetch_oi_history_extended(session, s, period, days) for s in symbols]
        
        # Fetch Funding rate
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        tasks_fr = [fetch_funding_rate(session, s, limit=MAX_FUNDING_LIMIT, start_time=start_time, end_time=end_time) for s in symbols]
        
        ois = await asyncio.gather(*tasks_oi, return_exceptions=True)
        frs = await asyncio.gather(*tasks_fr, return_exceptions=True)
    
    result = {}
    for sym, oi, fr in zip(symbols, ois, frs):
        # Convert SIRENUSDT -> SIREN/USDT
        if sym.endswith("USDT"):
            key = sym[:-4] + "/USDT"
        elif sym.endswith("USD"):
            key = sym[:-3] + "/USD"
        else:
            key = sym
            
        result[key] = {
            "oi": oi if isinstance(oi, pd.DataFrame) else pd.DataFrame(),
            "funding": fr if isinstance(fr, pd.DataFrame) else pd.DataFrame(),
        }
        
        oi_rows = len(result[key]["oi"])
        fr_rows = len(result[key]["funding"])
        logger.info(f"  OI/Funding collected for {key}: oi={oi_rows} rows, funding={fr_rows} rows")
        
    return result


async def collect_current_oi_batch(
    symbols: list,
) -> Dict[str, Optional[Dict]]:
    """
    Получить текущий Open Interest для списка символов.
    Быстрый способ получить OI без истории.
    
    Args:
        symbols: List of trading pair symbols (e.g., ["SIRENUSDT"])
    
    Returns:
        {"SIREN/USDT": {"openInterest": 12345.67, "timestamp": ...}, ...}
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_current_open_interest(session, s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output = {}
    for sym, data in zip(symbols, results):
        if sym.endswith("USDT"):
            key = sym[:-4] + "/USDT"
        else:
            key = sym
            
        if isinstance(data, dict):
            output[key] = data
        else:
            output[key] = None
            
    return output


# =============================================================================
# Convenience functions for synchronous usage
# =============================================================================

def get_oi_funding_sync(symbols: List[str], period: str = "15m", days: int = 30) -> Dict[str, Dict]:
    """
    Синхронная обертка для сбора OI/Funding.
    
    Args:
        symbols: List of symbols in Binance format (e.g., ["SIRENUSDT"])
        period: Time interval for OI
        days: Days of history
    
    Returns:
        Dict with OI and Funding data
    """
    return asyncio.run(collect_oi_funding_batch(symbols, period, days))


if __name__ == "__main__":
    # Test the collector
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    test_symbols = ["SIRENUSDT", "TAKEUSDT"]
    
    print(f"\n{'='*60}")
    print("Testing OI Collector")
    print(f"{'='*60}")
    
    result = asyncio.run(collect_oi_funding_batch(test_symbols, period="15m", days=7))
    
    print(f"\nResults:")
    for symbol, data in result.items():
        print(f"\n{symbol}:")
        print(f"  OI rows: {len(data['oi'])}")
        print(f"  Funding rows: {len(data['funding'])}")
        if not data['oi'].empty:
            print(f"  OI sample:\n{data['oi'].head(3)}")
        if not data['funding'].empty:
            print(f"  Funding sample:\n{data['funding'].head(3)}")
