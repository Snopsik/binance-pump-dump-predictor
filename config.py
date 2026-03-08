"""
Конфигурационный модуль для Binance Futures Pump/Dump Predictor.

Содержит все настраиваемые параметры пайплайна:
- GPU устройство для обучения
- Параметры API Binance
- Гиперпараметры модели
- Настройки feature engineering
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import os


class ModelType(Enum):
    """Доступные типы моделей для обучения."""
    CATBOOST = "catboost"
    XGBOOST = "xgboost"


class TimeFrame(Enum):
    """Поддерживаемые таймфреймы."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"


@dataclass
class GPUConfig:
    """
    Конфигурация GPU устройства.

    Важно: При наличии разных видеокарт (RTX 3060 + Tesla P100)
    код НЕ пытается объединить их в Multi-GPU,
    а обучает строго на выбранной карте.

    Attributes:
        device_id: ID устройства (0 или 1)
        use_gpu: Флаг использования GPU
        memory_limit: Лимит памяти GPU в MB (None = без ограничений)
    """
    device_id: int = 0
    use_gpu: bool = True
    memory_limit: Optional[int] = None

    def __post_init__(self):
        """Валидация параметров GPU."""
        if self.device_id not in [0, 1]:
            raise ValueError(f"GPU device_id должен быть 0 или 1, получено: {self.device_id}")
        if self.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)


@dataclass
class BinanceConfig:
    """
    Конфигурация подключения к Binance Futures API.

    Attributes:
        api_key: API ключ (можно передать через env)
        api_secret: API секрет (можно передать через env)
        rate_limit_requests: Макс. запросов в секунду
        rate_limit_orders: Макс. ордеров в секунду
        base_url: URL API Binance Futures
        recv_window: Окно времени для подписи (ms)
    """
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rate_limit_requests: int = 50  # Conservative limit
    rate_limit_orders: int = 10
    base_url: str = "https://fapi.binance.com"
    recv_window: int = 5000

    def __post_init__(self):
        """Загрузка ключей из переменных окружения если не переданы."""
        if self.api_key is None:
            self.api_key = os.getenv('BINANCE_API_KEY')
        if self.api_secret is None:
            self.api_secret = os.getenv('BINANCE_API_SECRET')


@dataclass
class FeatureConfig:
    """
    Конфигурация генерации признаков.

    Определяет окна для расчета индикаторов и групп признаков.
    """
    # Окна для скользящих расчетов
    short_window: int = 10
    medium_window: int = 30
    long_window: int = 60
    very_long_window: int = 120

    # Окна для volume anomaly
    volume_zscore_windows: List[int] = field(default_factory=lambda: [20, 60, 120])

    # Окна для CVD (Cumulative Volume Delta)
    cvd_windows: List[int] = field(default_factory=lambda: [10, 30])

    # Окно для корреляции с BTC
    btc_corr_window: int = 30

    # Окно для breakout детекции
    breakout_window: int = 20

    # Порог для определения пробоя
    breakout_threshold: float = 0.02  # 2% от high/low


@dataclass
class ModelHyperparams:
    """
    Гиперпараметры модели машинного обучения.

    Оптимизированы для несбалансированных данных (пампы/дампы < 1%).
    """
    # CatBoost параметры
    iterations: int = 2000
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 3.0
    min_data_in_leaf: int = 50
    random_strength: float = 1.0
    bagging_temperature: float = 0.8

    # Параметры для несбалансированных классов
    auto_class_weights: str = "Balanced"  # или "SqrtBalanced"
    scale_pos_weight: Optional[float] = None  # Будет вычислен автоматически

    # Early stopping
    early_stopping_rounds: int = 100
    eval_metric: str = "AUC"

    # Для TimeSeriesSplit
    n_splits: int = 5

    # Порог вероятности для алертов
    alert_threshold: float = 0.85


@dataclass
class PipelineConfig:
    """
    Главный конфигурационный класс, объединяющий все настройки.

    Example:
        >>> config = PipelineConfig(
        ...     gpu=GPUConfig(device_id=0),
        ...     model_type=ModelType.CATBOOST
        ... )
        >>> print(config.gpu.device_id)
        0
    """
    gpu: GPUConfig = field(default_factory=GPUConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model_params: ModelHyperparams = field(default_factory=ModelHyperparams)
    model_type: ModelType = ModelType.CATBOOST
    timeframe: TimeFrame = TimeFrame.MINUTE_1

    # Пути к данным
    labels_path: str = "labels.csv"
    data_cache_dir: str = "data_cache"
    model_save_path: str = "models/pump_predictor.cbm"

    # Параметры сбора данных
    history_days: int = 90  # Дней истории для обучения
    btc_symbol: str = "BTC/USDT"  # Символ для анализа режима рынка

    # Логирование
    log_level: str = "INFO"
    log_file: str = "logs/pipeline.log"

    def __post_init__(self):
        """Создание необходимых директорий."""
        os.makedirs(self.data_cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)


# =============================================================================
# Предустановленные конфигурации
# =============================================================================

def get_rtx3060_config() -> PipelineConfig:
    """
    Конфигурация для RTX 3060 (12GB VRAM).

    Returns:
        PipelineConfig с настройками для RTX 3060
    """
    return PipelineConfig(
        gpu=GPUConfig(device_id=0, use_gpu=True, memory_limit=10000),
        model_params=ModelHyperparams(
            depth=8,
            iterations=2000,
        )
    )


def get_tesla_p100_config() -> PipelineConfig:
    """
    Конфигурация для Tesla P100 (16GB VRAM).

    Returns:
        PipelineConfig с настройками для Tesla P100
    """
    return PipelineConfig(
        gpu=GPUConfig(device_id=1, use_gpu=True, memory_limit=14000),
        model_params=ModelHyperparams(
            depth=10,  # P100 может позволить больше
            iterations=3000,
        )
    )


def get_cpu_config() -> PipelineConfig:
    """
    Fallback конфигурация для CPU.

    Returns:
        PipelineConfig с отключенным GPU
    """
    return PipelineConfig(
        gpu=GPUConfig(device_id=0, use_gpu=False),
        model_params=ModelHyperparams(
            depth=6,  # Меньше для CPU
            iterations=1000,
        )
    )
