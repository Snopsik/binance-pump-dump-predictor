"""
Вспомогательные утилиты для пайплайна.

Содержит функции для:
- Валидации данных
- Проверки GPU доступности
- Утилит работы с временными рядами
- Экспорта результатов
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[int, Dict[str, Any]]:
    """
    Проверить доступность GPU устройств.

    Returns:
        Словарь с информацией о доступных GPU:
        {
            0: {'name': 'RTX 3060', 'memory': 12288, 'available': True},
            1: {'name': 'Tesla P100', 'memory': 16384, 'available': True}
        }
    """
    gpu_info = {}

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    idx = int(parts[0])
                    name = parts[1]
                    memory = int(float(parts[2]))  # MB

                    gpu_info[idx] = {
                        'name': name,
                        'memory_mb': memory,
                        'memory_gb': memory / 1024,
                        'available': True
                    }

    except Exception as e:
        logger.warning(f"Could not check GPU availability: {e}")

    return gpu_info


def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Валидация OHLCV DataFrame.

    Проверяет:
    - Наличие обязательных колонок
    - Отсутствие NaN в критичных полях
    - Корректность OHLC (high >= low, high >= open/close, etc.)
    - Положительные значения volume

    Args:
        df: DataFrame с OHLCV данными

    Returns:
        Кортеж (is_valid, list_of_errors)
    """
    errors = []

    # Проверяем обязательные колонки
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if errors:
        return False, errors

    # Проверяем NaN
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            errors.append(f"Column '{col}' has {nan_count} NaN values")

    # Проверяем корректность OHLC
    if (df['high'] < df['low']).any():
        errors.append("Some rows have high < low")

    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        errors.append("Some rows have high < open or high < close")

    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        errors.append("Some rows have low > open or low > close")

    # Проверяем положительность volume
    if (df['volume'] < 0).any():
        errors.append("Negative volume values found")

    return len(errors) == 0, errors


def validate_labels(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Валидация файла с метками.

    Args:
        df: DataFrame с метками

    Returns:
        Кортеж (is_valid, list_of_errors)
    """
    errors = []

    # Проверяем колонки
    required_cols = ['timestamp', 'symbol', 'label']
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if errors:
        return False, errors

    # Проверяем значения меток
    valid_labels = {1, -1, 0}
    invalid = set(df['label'].unique()) - valid_labels
    if invalid:
        errors.append(f"Invalid label values: {invalid}. Expected: {valid_labels}")

    # Проверяем NaN
    for col in required_cols:
        if df[col].isna().any():
            errors.append(f"Column '{col}' contains NaN values")

    return len(errors) == 0, errors


def calculate_label_distribution(labels: pd.Series) -> Dict[str, Any]:
    """
    Рассчитать распределение меток.

    Args:
        labels: Серия с метками

    Returns:
        Словарь со статистиками распределения
    """
    counts = labels.value_counts()
    total = len(labels)

    distribution = {
        'total': total,
        'counts': counts.to_dict(),
        'percentages': {
            'pump': counts.get(1, 0) / total * 100,
            'dump': counts.get(-1, 0) / total * 100,
            'neutral': counts.get(0, 0) / total * 100,
        },
        'imbalance_ratio': (counts.get(0, 0)) / (counts.get(1, 0) + counts.get(-1, 0) + 1)
    }

    return distribution


def infer_market_regime(
    btc_returns: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Определить режим рынка BTC (тренд/боковик).

    Args:
        btc_returns: Доходности BTC
        window: Окно для определения режима

    Returns:
        Series с режимами: 1 = uptrend, -1 = downtrend, 0 = sideways
    """
    # Скользящее среднее доходности
    rolling_mean = btc_returns.rolling(window=window).mean()
    # Скользящая волатильность
    rolling_std = btc_returns.rolling(window=window).std()

    # Тренд если |mean| > 0.5 * std
    regime = pd.Series(0, index=btc_returns.index)

    uptrend = rolling_mean > 0.5 * rolling_std
    downtrend = rolling_mean < -0.5 * rolling_std

    regime[uptrend] = 1
    regime[downtrend] = -1

    return regime


def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str = '5m'
) -> pd.DataFrame:
    """
    Пересэмплировать OHLCV данные в другой таймфрейм.

    Args:
        df: DataFrame с 1m данными
        target_timeframe: Целевой таймфрейм ('5m', '15m', '1h')

    Returns:
        DataFrame с пересэмплированными данными
    """
    df = df.copy()
    df.set_index('timestamp', inplace=True)

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'num_trades': 'sum',
        'taker_buy_base_volume': 'sum',
        'taker_buy_quote_volume': 'sum',
    }

    resampled = df.resample(target_timeframe).agg(agg_dict)
    resampled = resampled.dropna()
    resampled = resampled.reset_index()

    return resampled


def export_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: str,
    format: str = 'csv'
) -> None:
    """
    Экспортировать важность признаков в файл.

    Args:
        feature_importance: DataFrame с важностью признаков
        output_path: Путь для сохранения
        format: Формат ('csv', 'json', 'html')
    """
    if format == 'csv':
        feature_importance.to_csv(output_path, index=False)
    elif format == 'json':
        feature_importance.to_json(output_path, orient='records', indent=2)
    elif format == 'html':
        html = feature_importance.head(50).to_html(
            index=False,
            classes='table table-striped'
        )
        with open(output_path, 'w') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <title>Feature Importance</title>
</head>
<body class="container mt-4">
    <h1>Feature Importance Report</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
    {html}
</body>
</html>
            """)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Feature importance exported to {output_path}")


def create_backtest_report(
    predictions: pd.DataFrame,
    labels: pd.Series,
    output_path: str
) -> Dict[str, Any]:
    """
    Создать отчет о бэктесте предсказаний.

    Args:
        predictions: DataFrame с предсказаниями
        labels: Истинные метки
        output_path: Путь для сохранения отчета

    Returns:
        Словарь с метриками бэктеста
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )

    # Конвертируем в бинарную задачу для пампов
    y_true_pump = (labels == 1).astype(int)
    y_pred_pump = (predictions['predicted_label'] == 2).astype(int)  # 2 = памп

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(labels),
        'pump_metrics': {
            'accuracy': accuracy_score(y_true_pump, y_pred_pump),
            'precision': precision_score(y_true_pump, y_pred_pump, zero_division=0),
            'recall': recall_score(y_true_pump, y_pred_pump, zero_division=0),
            'f1': f1_score(y_true_pump, y_pred_pump, zero_division=0),
        },
        'confusion_matrix': confusion_matrix(y_true_pump, y_pred_pump).tolist(),
    }

    # Сохраняем отчет
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def get_time_until_next_minute() -> float:
    """
    Получить время в секундах до начала следующей минуты.

    Returns:
        Количество секунд до следующей минуты
    """
    now = datetime.utcnow()
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    return (next_minute - now).total_seconds()


def format_timedelta(td: timedelta) -> str:
    """
    Форматировать timedelta в читаемую строку.

    Args:
        td: Временной интервал

    Returns:
        Строка в формате "HH:MM:SS"
    """
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def ensure_directory(path: str) -> None:
    """
    Убедиться, что директория существует.

    Args:
        path: Путь к директории
    """
    os.makedirs(path, exist_ok=True)


def get_project_root() -> str:
    """
    Получить корневую директорию проекта.

    Returns:
        Абсолютный путь к корню проекта
    """
    return os.path.dirname(os.path.abspath(__file__))
