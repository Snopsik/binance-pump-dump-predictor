"""
Мульти-GPU конфигурация для параллельного обучения.

Архитектура:
- GPU 0 (RTX 3060): Модель детекции ПАМПОВ
- GPU 1 (Tesla P100): Модель детекции ДАМПОВ

Каждая модель - бинарный классификатор, обучаемый независимо.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import os
import multiprocessing as mp
from pathlib import Path


class ModelTarget(Enum):
    """Цель модели (что детектируем)."""
    PUMP = "pump"
    DUMP = "dump"


@dataclass
class DualGPUConfig:
    """
    Конфигурация для параллельной работы на двух GPU.

    GPU 0 (RTX 3060) -> Pump Detector
    GPU 1 (Tesla P100) -> Dump Detector

    Attributes:
        pump_gpu_id: GPU ID для модели пампов (default: 0 = RTX 3060)
        dump_gpu_id: GPU ID для модели дампов (default: 1 = Tesla P100)
        pump_model_params: Гиперпараметры для pump модели
        dump_model_params: Гиперпараметры для dump модели
    """
    # GPU назначения
    pump_gpu_id: int = 0  # RTX 3060
    dump_gpu_id: int = 1  # Tesla P100

    # Пути к моделям
    pump_model_path: str = "models/pump_detector.cbm"
    dump_model_path: str = "models/dump_detector.cbm"

    # Гиперпараметры для каждой модели
    pump_model_params: Dict[str, Any] = field(default_factory=lambda: {
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 8,  # RTX 3060 может позволить depth=8
        'l2_leaf_reg': 3.0,
        'min_data_in_leaf': 50,
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 100,
    })

    dump_model_params: Dict[str, Any] = field(default_factory=lambda: {
        'iterations': 3000,  # P100 мощнее, больше итераций
        'learning_rate': 0.02,
        'depth': 10,  # Tesla P100 может позволить depth=10
        'l2_leaf_reg': 3.0,
        'min_data_in_leaf': 30,
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 150,
    })

    # Пороги для алертов
    pump_alert_threshold: float = 0.85
    dump_alert_threshold: float = 0.85

    # Параметры обучения
    n_splits: int = 5
    history_days: int = 90

    # Пути
    labels_path: str = "labels.csv"
    data_cache_dir: str = "data_cache"
    log_dir: str = "logs"

    def __post_init__(self):
        """Создание директорий."""
        Path(self.data_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.pump_model_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.dump_model_path)).mkdir(parents=True, exist_ok=True)

    def get_cuda_device_for_target(self, target: ModelTarget) -> str:
        """
        Получить CUDA устройство для указанной цели.

        Args:
            target: ModelTarget.PUMP или ModelTarget.DUMP

        Returns:
            Строка с CUDA device (например, 'cuda:0')
        """
        if target == ModelTarget.PUMP:
            return f"cuda:{self.pump_gpu_id}"
        else:
            return f"cuda:{self.dump_gpu_id}"

    def get_gpu_id_for_target(self, target: ModelTarget) -> int:
        """
        Получить GPU ID для указанной цели.

        Args:
            target: ModelTarget.PUMP или ModelTarget.DUMP

        Returns:
            Integer GPU ID
        """
        return self.pump_gpu_id if target == ModelTarget.PUMP else self.dump_gpu_id

    def get_model_path_for_target(self, target: ModelTarget) -> str:
        """
        Получить путь к модели для указанной цели.

        Args:
            target: ModelTarget.PUMP или ModelTarget.DUMP

        Returns:
            Путь к файлу модели
        """
        return self.pump_model_path if target == ModelTarget.PUMP else self.dump_model_path

    def get_params_for_target(self, target: ModelTarget) -> Dict[str, Any]:
        """
        Получить гиперпараметры для указанной цели.

        Args:
            target: ModelTarget.PUMP или ModelTarget.DUMP

        Returns:
            Словарь с гиперпараметрами
        """
        return self.pump_model_params if target == ModelTarget.PUMP else self.dump_model_params

    def get_threshold_for_target(self, target: ModelTarget) -> float:
        """
        Получить порог вероятности для указанной цели.

        Args:
            target: ModelTarget.PUMP или ModelTarget.DUMP

        Returns:
            Порог вероятности
        """
        return self.pump_alert_threshold if target == ModelTarget.PUMP else self.dump_alert_threshold


@dataclass
class DualGPUMetrics:
    """
    Метрики для dual-GPU обучения.

    Хранит метрики отдельно для pump и dump моделей.
    """
    pump_metrics: Dict[str, float] = field(default_factory=dict)
    dump_metrics: Dict[str, float] = field(default_factory=dict)
    pump_training_time: float = 0.0
    dump_training_time: float = 0.0
    total_wall_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь."""
        return {
            'pump_metrics': self.pump_metrics,
            'dump_metrics': self.dump_metrics,
            'pump_training_time': self.pump_training_time,
            'dump_training_time': self.dump_training_time,
            'total_wall_time': self.total_wall_time,
            'speedup_factor': (
                self.pump_training_time + self.dump_training_time
            ) / max(self.total_wall_time, 0.001)
        }


def print_gpu_assignment(config: DualGPUConfig) -> None:
    """
    Вывести назначение GPU устройств.

    Args:
        config: Конфигурация DualGPU
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                    DUAL-GPU PARALLEL TRAINING                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   ┌─────────────────────┐         ┌─────────────────────┐            ║
║   │   GPU 0: RTX 3060   │         │  GPU 1: Tesla P100  │            ║
║   │                     │         │                     │            ║
║   │   📈 PUMP DETECTOR  │         │  📉 DUMP DETECTOR   │            ║
║   │                     │         │                     │            ║
║   │   - Binary Model    │         │   - Binary Model    │            ║
║   │   - Depth: 8        │         │   - Depth: 10       │            ║
║   │   - Iterations: 2K  │         │   - Iterations: 3K  │            ║
║   └─────────────────────┘         └─────────────────────┘            ║
║                                                                       ║
║   Обе модели обучаются ПАРАЛЛЕЛЬНО в отдельных процессах!            ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
