"""
Модуль обучения модели с GPU поддержкой.

Ключевые особенности:
- CatBoostClassifier/XGBoost с GPU ускорением
- TimeSeriesSplit для корректной валидации временных рядов
- Автоматическая балансировка классов
- Feature Importance анализ
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV

from config import PipelineConfig, ModelType, GPUConfig

logger = logging.getLogger(__name__)

PURGE_GAP = 64


def _purged_time_series_split(n_samples: int,
                              n_splits: int,
                              purge_gap: int = PURGE_GAP):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(np.arange(n_samples)):
        if purge_gap > 0 and len(train_idx) > purge_gap:
            train_idx = train_idx[:-purge_gap]
        yield train_idx, val_idx


@dataclass
class TrainingResult:
    """
    Результаты обучения модели.

    Attributes:
        model: Обученная модель
        metrics: Словарь метрик на валидации
        feature_importance: DataFrame с важностью признаков
        best_iteration: Лучшая итерация (early stopping)
        training_time: Время обучения в секундах
    """
    model: Any
    metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    best_iteration: int
    training_time: float


class PumpDumpModel:
    """
    Класс для обучения и управления моделью предсказания пампов/дампов.

    Поддерживает CatBoost и XGBoost с GPU ускорением.

    Example:
        >>> model = PumpDumpModel(config)
        >>> result = model.train(X, y)
        >>> model.save("model.cbm")
    """

    def __init__(self, config: PipelineConfig):
        """
        Инициализация модели.

        Args:
            config: Конфигурация пайплайна
        """
        self.config = config
        self.model_type = config.model_type
        self.gpu_config = config.gpu
        self.model_params = config.model_params

        self._model: Optional[Any] = None
        self._feature_names: Optional[List[str]] = None

    def _get_catboost_params(self) -> Dict[str, Any]:
        """
        Получить параметры для CatBoostClassifier.

        Важно: Использует task_type='GPU' и devices=str(device_id)
        для обучения на конкретной GPU (не Multi-GPU).

        Returns:
            Словарь параметров CatBoost
        """
        params = {
            'iterations': self.model_params.iterations,
            'learning_rate': self.model_params.learning_rate,
            'depth': self.model_params.depth,
            'l2_leaf_reg': self.model_params.l2_leaf_reg,
            'min_data_in_leaf': self.model_params.min_data_in_leaf,
            'random_strength': self.model_params.random_strength,
            'bagging_temperature': self.model_params.bagging_temperature,
            'auto_class_weights': self.model_params.auto_class_weights,
            'eval_metric': self.model_params.eval_metric,
            'early_stopping_rounds': self.model_params.early_stopping_rounds,
            'random_seed': 42,
            'verbose': 100,
            'allow_writing_files': False,
        }

        # GPU конфигурация - КРИТИЧЕСКИ ВАЖНО
        if self.gpu_config.use_gpu:
            params['task_type'] = 'GPU'
            # Используем только указанное устройство
            # Не '0,1' что вызвало бы Multi-GPU
            params['devices'] = str(self.gpu_config.device_id)

            if self.gpu_config.memory_limit:
                params[
                    'max_ctr_complexity'] = 4  # Уменьшаем сложность для экономии памяти

        return params

    def _get_xgboost_params(self) -> Dict[str, Any]:
        """
        Получить параметры для XGBoost.

        Returns:
            Словарь параметров XGBoost
        """
        params = {
            'n_estimators': self.model_params.iterations,
            'learning_rate': self.model_params.learning_rate,
            'max_depth': self.model_params.depth,
            'reg_lambda': self.model_params.l2_leaf_reg,
            'min_child_weight': self.model_params.min_data_in_leaf,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': -1,
        }

        # GPU конфигурация для XGBoost
        if self.gpu_config.use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = f'cuda:{self.gpu_config.device_id}'

        return params

    def _calculate_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """
        Рассчитать веса классов для несбалансированных данных.

        Поскольку пампы/дампы составляют < 1% данных,
        критически важно использовать балансировку.

        Args:
            y: Серия с метками классов

        Returns:
            Словарь {class_label: weight}
        """
        class_counts = y.value_counts()
        total = len(y)

        weights = {}
        for label in class_counts.index:
            # Weight = total / (n_classes * count)
            weights[label] = total / (len(class_counts) * class_counts[label])

        logger.info(f"Calculated class weights: {weights}")
        return weights

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_groups: Optional[Dict[str,
                                      List[str]]] = None) -> TrainingResult:
        """
        Обучить модель с TimeSeriesSplit валидацией.

        Важно: Используется TimeSeriesSplit, а не KFold,
        так как данные имеют временную структуру.

        Args:
            X: DataFrame с признаками
            y: Серия с метками (1=памп, -1=дамп, 0=нейтрально)
            feature_groups: Группы признаков для анализа важности

        Returns:
            TrainingResult с результатами обучения
        """
        import time
        start_time = time.time()

        zero_var_cols = X.columns[X.std() == 0].tolist()
        if zero_var_cols:
            logger.info(
                f"Dropping {len(zero_var_cols)} zero-variance features: {zero_var_cols}"
            )
            X = X.drop(columns=zero_var_cols)

        self._feature_names = X.columns.tolist()

        y_transformed = y.map({-1: 0, 0: 1, 1: 2})

        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")

        # Инициализируем модель в зависимости от типа
        if self.model_type == ModelType.CATBOOST:
            result = self._train_catboost(X, y_transformed)
        else:
            result = self._train_xgboost(X, y_transformed)

        result.training_time = time.time() - start_time

        # Анализируем важность признаков по группам
        if feature_groups:
            self._analyze_feature_groups(result.feature_importance,
                                         feature_groups)

        return result

    def _train_catboost(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """
        Обучение CatBoostClassifier.

        Args:
            X: Признаки
            y: Трансформированные метки

        Returns:
            TrainingResult
        """
        from catboost import CatBoostClassifier, Pool

        params = self._get_catboost_params()

        logger.info(f"CatBoost parameters: {params}")
        logger.info(f"Using GPU device: {self.gpu_config.device_id}")

        all_metrics = []
        best_models = []

        for fold, (train_idx, val_idx) in enumerate(
                _purged_time_series_split(len(X),
                                          self.model_params.n_splits,
                                          purge_gap=PURGE_GAP)):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold + 1}/{self.model_params.n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
            logger.info(
                f"Train target distribution:\n{y_train.value_counts()}")

            # Создаем модель
            model = CatBoostClassifier(**params)

            # Обучаем с early stopping
            model.fit(X_train,
                      y_train,
                      eval_set=(X_val, y_val),
                      verbose=100,
                      use_best_model=True)

            # Предсказания на валидации
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            # Метрики
            metrics = self._calculate_metrics(y_val, y_pred, y_proba)
            all_metrics.append(metrics)

            logger.info(
                f"Fold {fold + 1} metrics: AUC={metrics['auc_macro']:.4f}, "
                f"F1_macro={metrics['f1_macro']:.4f}")

            best_models.append(model)

        # Выбираем лучшую модель по среднему AUC
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }

        best_fold = np.argmax([m['auc_macro'] for m in all_metrics])
        self._model = best_models[best_fold]

        logger.info(f"\n{'='*50}")
        logger.info(f"Average metrics across folds:")
        for key, value in avg_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature':
            self._feature_names,
            'importance':
            self._model.feature_importances_ if hasattr(
                self._model, 'feature_importances_') else []
        }).sort_values('importance', ascending=False)

        return TrainingResult(
            model=self._model,
            metrics=avg_metrics,
            feature_importance=feature_importance,
            best_iteration=self._model.best_iteration_ if hasattr(
                self._model, 'best_iteration_') else 0,
            training_time=0.0  # Will be set by caller
        )

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """
        Обучение XGBoost.

        Args:
            X: Признаки
            y: Трансформированные метки

        Returns:
            TrainingResult
        """
        from xgboost import XGBClassifier

        params = self._get_xgboost_params()

        # Для мультикласса
        params['num_class'] = 3
        params['objective'] = 'multi:softprob'

        logger.info(f"XGBoost parameters: {params}")

        all_metrics = []
        best_models = []
        best_iterations = []

        for fold, (train_idx, val_idx) in enumerate(
                _purged_time_series_split(len(X),
                                          self.model_params.n_splits,
                                          purge_gap=PURGE_GAP)):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold + 1}/{self.model_params.n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(**params)

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            metrics = self._calculate_metrics(y_val, y_pred, y_proba)
            all_metrics.append(metrics)

            best_models.append(model)
            best_iterations.append(model.best_iteration)

        # Средние метрики
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }

        best_fold = np.argmax([m['auc_macro'] for m in all_metrics])
        self._model = best_models[best_fold]

        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature':
            self._feature_names,
            'importance':
            self._model.feature_importances_ if hasattr(
                self._model, 'feature_importances_') else []
        }).sort_values('importance', ascending=False)

        return TrainingResult(model=self._model,
                              metrics=avg_metrics,
                              feature_importance=feature_importance,
                              best_iteration=best_iterations[best_fold]
                              if best_iterations else 0,
                              training_time=0.0)

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                           y_proba: np.ndarray) -> Dict[str, float]:
        """
        Рассчитать метрики классификации.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            y_proba: Вероятности классов

        Returns:
            Словарь метрик
        """
        # AUC для мультикласса (One-vs-Rest)
        try:
            auc_macro = roc_auc_score(y_true,
                                      y_proba,
                                      multi_class='ovr',
                                      average='macro')
        except Exception:
            auc_macro = 0.0

        # F1-score
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # Precision/Recall для каждого класса
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0)

        # Класс 0 = дамп, Класс 2 = памп
        # Нам важны именно эти классы
        metrics = {
            'auc_macro':
            auc_macro,
            'f1_macro':
            f1_macro,
            'f1_weighted':
            f1_weighted,
            'precision_dump':
            precision[0]
            if isinstance(precision,
                          (list, np.ndarray)) and len(precision) > 0 else 0,
            'recall_dump':
            recall[0]
            if isinstance(recall,
                          (list, np.ndarray)) and len(recall) > 0 else 0,
            'f1_dump':
            f1[0] if isinstance(f1, (list, np.ndarray)) and len(f1) > 0 else 0,
            'precision_pump':
            precision[2]
            if isinstance(precision,
                          (list, np.ndarray)) and len(precision) > 2 else 0,
            'recall_pump':
            recall[2] if isinstance(recall, (list, np.ndarray))
            and len(recall) > 2 else 0,
            'f1_pump':
            f1[2] if isinstance(f1, (list, np.ndarray)) and len(f1) > 2 else 0,
        }

        return metrics

    def _analyze_feature_groups(self, feature_importance: pd.DataFrame,
                                feature_groups: Dict[str, List[str]]) -> None:
        """
        Анализ важности групп признаков.

        Позволяет убедиться, что Trade Flow и Volume Anomaly
        действительно доминируют в модели.

        Args:
            feature_importance: DataFrame с важностью признаков
            feature_groups: Группы признаков
        """
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE IMPORTANCE BY GROUPS")
        logger.info("=" * 60)

        # Убираем суффикс _lag1 для сопоставления
        feature_importance['feature_base'] = feature_importance[
            'feature'].str.replace('_lag1', '')

        group_importance = {}
        for group_name, features in feature_groups.items():
            if not features:
                continue

            # Ищем признаки группы в топе
            group_df = feature_importance[
                feature_importance['feature_base'].isin(features)]
            total_importance = group_df['importance'].sum()
            avg_importance = group_df['importance'].mean() if len(
                group_df) > 0 else 0

            group_importance[group_name] = {
                'total': total_importance,
                'average': avg_importance,
                'count': len(group_df)
            }

        # Сортируем по общей важности
        sorted_groups = sorted(group_importance.items(),
                               key=lambda x: x[1]['total'],
                               reverse=True)

        for group_name, stats in sorted_groups:
            logger.info(f"{group_name}:")
            logger.info(f"  Total importance: {stats['total']:.2f}")
            logger.info(f"  Average importance: {stats['average']:.4f}")
            logger.info(f"  Feature count: {stats['count']}")

        # Топ-20 признаков
        logger.info("\n" + "=" * 60)
        logger.info("TOP 20 FEATURES")
        logger.info("=" * 60)
        for idx, row in feature_importance.head(20).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказать классы для новых данных.

        Args:
            X: DataFrame с признаками

        Returns:
            Массив предсказанных классов (0=дамп, 1=нейтрально, 2=памп)
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказать вероятности классов.

        Args:
            X: DataFrame с признаками

        Returns:
            Массив вероятностей shape (n_samples, 3)
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self._model.predict_proba(X)

    def get_pump_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить вероятность пампа (класс 2).

        Args:
            X: DataFrame с признаками

        Returns:
            Массив вероятностей пампа
        """
        proba = self.predict_proba(X)
        return proba[:, 2]  # Класс 2 = памп

    def get_dump_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить вероятность дампа (класс 0).

        Args:
            X: DataFrame с признаками

        Returns:
            Массив вероятностей дампа
        """
        proba = self.predict_proba(X)
        return proba[:, 0]  # Класс 0 = дамп

    def save(self, path: str) -> None:
        """
        Сохранить модель в файл.

        Args:
            path: Путь для сохранения
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if self.model_type == ModelType.CATBOOST:
            self._model.save_model(path)
        else:
            self._model.save_model(path)

        # Сохраняем метаданные
        metadata = {
            'model_type': self.model_type.value,
            'feature_names': self._feature_names,
            'gpu_device_id': self.gpu_config.device_id,
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = Path(path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")
        logger.info(f"Metadata saved to {metadata_path}")

    def load(self, path: str) -> None:
        """
        Загрузить модель из файла.

        Args:
            path: Путь к файлу модели
        """
        # Загружаем метаданные
        metadata_path = Path(path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self._feature_names = metadata.get('feature_names')

        # Загружаем модель
        if self.model_type == ModelType.CATBOOST:
            from catboost import CatBoostClassifier
            self._model = CatBoostClassifier()
            self._model.load_model(path)
        else:
            from xgboost import XGBClassifier
            self._model = XGBClassifier()
            self._model.load_model(path)

        logger.info(f"Model loaded from {path}")


def train_model(
        config: PipelineConfig,
        X: pd.DataFrame,
        y: pd.Series,
        feature_groups: Optional[Dict[str,
                                      List[str]]] = None) -> TrainingResult:
    """
    Главная функция для обучения модели.

    Args:
        config: Конфигурация пайплайна
        X: Признаки
        y: Таргет
        feature_groups: Группы признаков

    Returns:
        TrainingResult
    """
    model = PumpDumpModel(config)
    result = model.train(X, y, feature_groups)
    model.save(config.model_save_path)

    return result
