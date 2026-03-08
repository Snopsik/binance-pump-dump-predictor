"""
Параллельное обучение на двух GPU.

Архитектура:
- GPU 0 (RTX 3060): Обучает PUMP детектор
- GPU 1 (Tesla P100): Обучает DUMP детектор

Использует multiprocessing для изоляции GPU контекстов.
Каждый процесс работает с одной GPU без Multi-GPU overhead.
"""

import os
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import traceback
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from dual_gpu_config import DualGPUConfig, ModelTarget, DualGPUMetrics
from oi_collector import collect_oi_funding_batch
# Настройка логирования для процессов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def set_gpu_environment(gpu_id: int) -> None:
    """
    Установить переменную окружения CUDA_VISIBLE_DEVICES для процесса.

    КРИТИЧЕСКИ ВАЖНО: Это ограничивает процесс одной GPU,
    предотвращая попытки Multi-GPU.

    Args:
        gpu_id: ID GPU устройства
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def prepare_binary_target(y: pd.Series, target: ModelTarget) -> pd.Series:
    """
    Подготовить бинарный таргет для указанной цели.

    Для PUMP: y = 1 если label == 1, иначе 0
    Для DUMP: y = 1 если label == -1, иначе 0

    Args:
        y: Исходные метки (1=памп, -1=дамп, 0=нейтрально)
        target: Целевое событие (PUMP или DUMP)

    Returns:
        Бинарный таргет
    """
    if target == ModelTarget.PUMP:
        return (y == 1).astype(int)
    else:  # DUMP
        return (y == -1).astype(int)


PURGE_GAP = 64


def _purged_time_series_split(n_samples: int, n_splits: int, purge_gap: int = PURGE_GAP):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(np.arange(n_samples)):
        if purge_gap > 0 and len(train_idx) > purge_gap:
            train_idx = train_idx[:-purge_gap]
        yield train_idx, val_idx


def train_single_model_process(
    gpu_id: int,
    target: ModelTarget,
    X_path: str,
    y_path: str,
    model_path: str,
    model_params: Dict[str, Any],
    n_splits: int,
    result_queue: Queue,
    feature_names: List[str],
    use_gpu: bool = True
) -> None:
    """
    Функция для обучения одной модели в отдельном процессе.
    Uses purged time series split to avoid look-ahead bias.
    """
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        device_name = f"GPU {gpu_id}"
    else:
        device_name = "CPU"

    process_logger = logging.getLogger(f"{target.value}_process")
    process_logger.info(f"Starting {target.value.upper()} training on {device_name}")

    start_time = time.time()

    try:
        from catboost import CatBoostClassifier
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve

        X = pd.read_parquet(X_path)
        y = pd.read_parquet(y_path)['label']
        y_binary = prepare_binary_target(y, target)

        process_logger.info(f"Data loaded: X shape={X.shape}, positive ratio={y_binary.mean():.4f}")

        zero_var_cols = X.columns[X.std() == 0].tolist()
        if zero_var_cols:
            process_logger.info(f"Dropping {len(zero_var_cols)} zero-variance columns")
            X = X.drop(columns=zero_var_cols)
            feature_names = [f for f in feature_names if f not in zero_var_cols]

        params = {
            **model_params,
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': 100,
            'allow_writing_files': False,
        }
        if use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = '0'
        else:
            params['task_type'] = 'CPU'
            params['depth'] = min(params.get('depth', 8), 6)
            params['iterations'] = min(params.get('iterations', 2000), 500)

        all_metrics = []
        best_models = []
        best_scores = []
        
        last_X_val = None
        last_y_val = None
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(
            _purged_time_series_split(len(X), n_splits, purge_gap=PURGE_GAP)
        ):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_binary.iloc[train_idx], y_binary.iloc[val_idx]

            if y_val.sum() == 0:
                process_logger.warning(f"Fold {fold + 1}: No positives in validation, skip.")
                continue

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100, use_best_model=True)

            # Метрики при пороге 0.5
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            try:
                auc = roc_auc_score(y_val, y_proba)
            except:
                auc = 0.5
            
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            all_metrics.append({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1})
            best_models.append(model)
            best_scores.append(auc)
            
            # Запоминаем последний фолд
            last_X_val = X_val
            last_y_val = y_val
            
            process_logger.info(f"Fold {fold + 1} - AUC: {auc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")

        # Выбор лучшей модели
        if not best_models:
             raise ValueError("No valid folds trained")
             
        best_fold = np.argmax(best_scores)
        best_model = best_models[best_fold]

        # --- НОВОЕ: ПОИСК ОПТИМАЛЬНОГО ПОРОГА ---
        # Мы хотим найти порог, где Precision >= 20% (Target)
        TARGET_PRECISION = 0.20
        
        if last_y_val is not None and last_y_val.sum() > 0:
            proba_val = best_model.predict_proba(last_X_val)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(last_y_val, proba_val)
            
            # precisions длиннее thresholds на 1, выравниваем
            precisions = precisions[:-1]
            recalls = recalls[:-1]
            
            # Ищем индексы, где precision >= TARGET
            candidates = np.where(precisions >= TARGET_PRECISION)[0]
            
            if len(candidates) > 0:
                # Берем тот, где максимальный Recall
                best_idx = candidates[np.argmax(recalls[candidates])]
                opt_method = f"Target {TARGET_PRECISION:.0%}"
            else:
                # Если не нашли 20%, берем максимальный F1
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
                best_idx = np.argmax(f1_scores)
                opt_method = "Max F1"
            
            optimal_threshold = thresholds[best_idx]
            opt_prec = precisions[best_idx]
            opt_rec = recalls[best_idx]
            
            # ЛОГ ВЫВОДА (его ищем в консоли)
            process_logger.info(f"=" * 40)
            process_logger.info(f"OPTIMAL THRESHOLD ({opt_method}): {optimal_threshold:.4f}")
            process_logger.info(f"Expected Precision: {opt_prec:.2%}, Recall: {opt_rec:.2%}")
            process_logger.info(f"=" * 40)
        else:
            optimal_threshold = 0.5
            opt_prec = 0.0
            opt_rec = 0.0
            process_logger.warning("Could not optimize threshold, using 0.5")

        # Средние метрики
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
        avg_metrics['optimal_threshold'] = float(optimal_threshold)
        avg_metrics['opt_precision'] = float(opt_prec)
        avg_metrics['opt_recall'] = float(opt_rec)

        training_time = time.time() - start_time
        best_model.save_model(model_path)
        
        # Feature Importance
        fi_path = model_path.replace('.cbm', '_feature_importance.csv')
        pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False).to_csv(fi_path, index=False)

        result_queue.put({
            'target': target.value,
            'success': True,
            'metrics': avg_metrics,
            'training_time': training_time,
            'best_fold': int(best_fold),
            'error': None
        })

    except Exception as e:
        error_msg = f"Error in {target.value} training: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        result_queue.put({
            'target': target.value,
            'success': False,
            'metrics': {},
            'training_time': time.time() - start_time,
            'best_fold': -1,
            'error': str(e)
        })


class DualGPUTrainer:
    """
    Класс для параллельного обучения двух моделей на разных GPU или CPU.

    Запускает два отдельных процесса:
    - Процесс 1: PUMP детектор на GPU 0 (RTX 3060)
    - Процесс 2: DUMP детектор на GPU 1 (Tesla P100)

    Example:
        >>> config = DualGPUConfig()
        >>> trainer = DualGPUTrainer(config)
        >>> metrics = trainer.train_parallel(X, y)
    """

    def __init__(self, config: DualGPUConfig, use_gpu: bool = True):
        """
        Инициализация тренера.

        Args:
            config: Конфигурация DualGPU
            use_gpu: Использовать GPU (если False, работает на CPU)
        """
        self.config = config
        self.use_gpu = use_gpu
        self._temp_dir = "/tmp/dual_gpu_training"

        # Создаем временную директорию
        os.makedirs(self._temp_dir, exist_ok=True)

    def train_parallel(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> DualGPUMetrics:
        """
        Параллельно обучить обе модели.

        Args:
            X: DataFrame с признаками
            y: Серия с метками (1=памп, -1=дамп, 0=нейтрально)
            feature_names: Список названий признаков

        Returns:
            DualGPUMetrics с метриками обеих моделей
        """
        from dual_gpu_config import print_gpu_assignment
        print_gpu_assignment(self.config)

        logger.info("Preparing data for parallel training...")

        # Сохраняем данные во временные файлы для передачи в процессы
        X_path = os.path.join(self._temp_dir, "X.parquet")
        y_path = os.path.join(self._temp_dir, "y.parquet")

        X.to_parquet(X_path, index=False)
        y.to_frame('label').to_parquet(y_path, index=False)

        if feature_names is None:
            feature_names = X.columns.tolist()

        # Очередь для результатов
        result_queue = Queue()

        # Создаем процессы
        pump_process = Process(
            target=train_single_model_process,
            kwargs={
                'gpu_id': self.config.pump_gpu_id,
                'target': ModelTarget.PUMP,
                'X_path': X_path,
                'y_path': y_path,
                'model_path': self.config.pump_model_path,
                'model_params': self.config.pump_model_params,
                'n_splits': self.config.n_splits,
                'result_queue': result_queue,
                'feature_names': feature_names,
                'use_gpu': self.use_gpu
            }
        )

        dump_process = Process(
            target=train_single_model_process,
            kwargs={
                'gpu_id': self.config.dump_gpu_id,
                'target': ModelTarget.DUMP,
                'X_path': X_path,
                'y_path': y_path,
                'model_path': self.config.dump_model_path,
                'model_params': self.config.dump_model_params,
                'n_splits': self.config.n_splits,
                'result_queue': result_queue,
                'feature_names': feature_names,
                'use_gpu': self.use_gpu
            }
        )

        logger.info("Starting parallel training...")
        wall_start = time.time()

        # Запускаем оба процесса
        pump_process.start()
        dump_process.start()

        device_name = "GPU" if self.use_gpu else "CPU"
        logger.info(f"PUMP process started ({device_name} {self.config.pump_gpu_id if self.use_gpu else ''})")
        logger.info(f"DUMP process started ({device_name} {self.config.dump_gpu_id if self.use_gpu else ''})")

        # Ждем завершения
        pump_process.join()
        dump_process.join()

        wall_time = time.time() - wall_start

        logger.info(f"All processes completed in {wall_time:.2f}s")

        # Собираем результаты
        results = {}
        while not result_queue.empty():
            result = result_queue.get()
            results[result['target']] = result

        # Формируем метрики
        metrics = DualGPUMetrics()
        metrics.total_wall_time = wall_time

        if 'pump' in results:
            pump_result = results['pump']
            if pump_result['success']:
                metrics.pump_metrics = pump_result['metrics']
                metrics.pump_training_time = pump_result['training_time']
                logger.info(f"PUMP model trained successfully: AUC={pump_result['metrics'].get('auc', 0):.4f}")
            else:
                logger.error(f"PUMP model failed: {pump_result['error']}")

        if 'dump' in results:
            dump_result = results['dump']
            if dump_result['success']:
                metrics.dump_metrics = dump_result['metrics']
                metrics.dump_training_time = dump_result['training_time']
                logger.info(f"DUMP model trained successfully: AUC={dump_result['metrics'].get('auc', 0):.4f}")
            else:
                logger.error(f"DUMP model failed: {dump_result['error']}")

        # Выводим статистику
        self._print_training_summary(metrics, results)

        # Очищаем временные файлы
        self._cleanup()

        return metrics

    def train_sequential(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> DualGPUMetrics:
        """
        Последовательно обучить обе модели (для сравнения).

        Args:
            X: DataFrame с признаками
            y: Серия с метками
            feature_names: Список названий признаков

        Returns:
            DualGPUMetrics
        """
        logger.info("Starting SEQUENTIAL training (for comparison)...")

        # Сохраняем данные
        X_path = os.path.join(self._temp_dir, "X.parquet")
        y_path = os.path.join(self._temp_dir, "y.parquet")
        X.to_parquet(X_path, index=False)
        y.to_frame('label').to_parquet(y_path, index=False)

        if feature_names is None:
            feature_names = X.columns.tolist()

        result_queue = Queue()
        metrics = DualGPUMetrics()
        wall_start = time.time()

        # Обучаем PUMP модель
        train_single_model_process(
            gpu_id=self.config.pump_gpu_id,
            target=ModelTarget.PUMP,
            X_path=X_path,
            y_path=y_path,
            model_path=self.config.pump_model_path,
            model_params=self.config.pump_model_params,
            n_splits=self.config.n_splits,
            result_queue=result_queue,
            feature_names=feature_names
        )

        pump_result = result_queue.get()
        if pump_result['success']:
            metrics.pump_metrics = pump_result['metrics']
            metrics.pump_training_time = pump_result['training_time']

        # Обучаем DUMP модель
        train_single_model_process(
            gpu_id=self.config.dump_gpu_id,
            target=ModelTarget.DUMP,
            X_path=X_path,
            y_path=y_path,
            model_path=self.config.dump_model_path,
            model_params=self.config.dump_model_params,
            n_splits=self.config.n_splits,
            result_queue=result_queue,
            feature_names=feature_names
        )

        dump_result = result_queue.get()
        if dump_result['success']:
            metrics.dump_metrics = dump_result['metrics']
            metrics.dump_training_time = dump_result['training_time']

        metrics.total_wall_time = time.time() - wall_start

        logger.info(f"Sequential training completed in {metrics.total_wall_time:.2f}s")

        self._cleanup()
        return metrics

    def _print_training_summary(
        self,
        metrics: DualGPUMetrics,
        results: Dict[str, Any]
    ) -> None:
        """
        Вывести сводку по обучению.

        Args:
            metrics: Метрики обучения
            results: Результаты из процессов
        """
        print("\n" + "="*70)
        print("DUAL-GPU TRAINING SUMMARY")
        print("="*70)

        print(f"\nWall time: {metrics.total_wall_time:.2f}s")
        print(f"PUMP training time: {metrics.pump_training_time:.2f}s (GPU {self.config.pump_gpu_id})")
        print(f"DUMP training time: {metrics.dump_training_time:.2f}s (GPU {self.config.dump_gpu_id})")

        theoretical_time = metrics.pump_training_time + metrics.dump_training_time
        speedup = theoretical_time / max(metrics.total_wall_time, 0.001)
        print(f"\nSpeedup factor: {speedup:.2f}x (parallel vs sequential)")

        print("\n" + "-"*70)
        print("PUMP DETECTOR METRICS (GPU 0 - RTX 3060)")
        print("-"*70)
        for key, value in metrics.pump_metrics.items():
            print(f"  {key}: {value:.4f}")

        print("\n" + "-"*70)
        print("DUMP DETECTOR METRICS (GPU 1 - Tesla P100)")
        print("-"*70)
        for key, value in metrics.dump_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Топ признаки
        if 'pump' in results and results['pump'].get('feature_importance'):
            print("\n" + "-"*70)
            print("TOP 10 PUMP FEATURES")
            print("-"*70)
            for i, fi in enumerate(results['pump']['feature_importance'][:10], 1):
                print(f"  {i}. {fi['feature']}: {fi['importance']:.4f}")

        if 'dump' in results and results['dump'].get('feature_importance'):
            print("\n" + "-"*70)
            print("TOP 10 DUMP FEATURES")
            print("-"*70)
            for i, fi in enumerate(results['dump']['feature_importance'][:10], 1):
                print(f"  {i}. {fi['feature']}: {fi['importance']:.4f}")

        print("\n" + "="*70)
        print(f"Models saved:")
        print(f"  PUMP: {self.config.pump_model_path}")
        print(f"  DUMP: {self.config.dump_model_path}")
        print("="*70 + "\n")

    def _cleanup(self) -> None:
        """Удалить временные файлы."""
        import shutil
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)


def run_dual_training(
    config: DualGPUConfig,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Optional[List[str]] = None,
    sequential: bool = False,
    use_gpu: bool = True
) -> DualGPUMetrics:
    """
    Главная функция для запуска параллельного обучения.

    Args:
        config: Конфигурация DualGPU
        X: Признаки
        y: Таргет
        feature_names: Названия признаков
        sequential: Если True, обучает последовательно (для сравнения)
        use_gpu: Использовать GPU (если False, работает на CPU)

    Returns:
        DualGPUMetrics
    """
    trainer = DualGPUTrainer(config, use_gpu=use_gpu)

    if sequential:
        return trainer.train_sequential(X, y, feature_names)
    else:
        return trainer.train_parallel(X, y, feature_names)
