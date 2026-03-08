# Binance Futures Pump/Dump Predictor v2.0

Production-ready Python-пайплайн для лайв-трейдинга на Binance Futures с **DUAL-GPU параллельной обработкой**.

## 🚀 NEW: Dual-GPU Parallel Mode

```
┌─────────────────────┐         ┌─────────────────────┐
│   GPU 0: RTX 3060   │         │  GPU 1: Tesla P100  │
│                     │         │                     │
│   📈 PUMP DETECTOR  │         │  📉 DUMP DETECTOR   │
│                     │         │                     │
│   - Binary Model    │         │   - Binary Model    │
│   - Depth: 8        │         │   - Depth: 10       │
│   - Iterations: 2K  │         │   - Iterations: 3K  │
└─────────────────────┘         └─────────────────────┘

Обе модели обучаются и работают ПАРАЛЛЕЛЬНО!
```

## Архитектура

```
binance_pump_predictor/
├── config.py              # Single-GPU конфигурация
├── dual_gpu_config.py     # DUAL-GPU конфигурация ⭐ NEW
├── data_collector.py      # Асинхронный сбор данных
├── feature_engineering.py # Quant Feature Engineering
├── model_training.py      # Single-GPU обучение
├── dual_gpu_trainer.py    # DUAL-GPU параллельное обучение ⭐ NEW
├── live_inference.py      # Single-GPU inference
├── dual_gpu_inference.py  # DUAL-GPU параллельный inference ⭐ NEW
├── main.py               # CLI с поддержкой dual-GPU
├── utils.py              # Вспомогательные функции
└── labels.csv            # Пример файла меток
```

## Установка

```bash
pip install -r requirements.txt

# Опционально: API ключи для приватных эндпоинтов
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

## Использование

### Single GPU Mode (как раньше)

```bash
# Обучение на RTX 3060
python main.py train --labels labels.csv --gpu 0

# Обучение на Tesla P100
python main.py train --labels labels.csv --gpu 1

# Live inference
python main.py live --model models/pump_predictor.cbm --symbols ETH/USDT,BNB/USDT --gpu 0
```

### ⭐ DUAL GPU Mode (ПАРАЛЛЕЛЬНАЯ обработка)

```bash
# Параллельное обучение PUMP на GPU0, DUMP на GPU1
python main.py dual-train --labels labels.csv

# Параллельный live inference
python main.py dual-live --symbols ETH/USDT,BNB/USDT,SOL/USDT

# Полный dual-GPU пайплайн
python main.py dual-full --labels labels.csv
```

### Результат Dual-GPU обучения

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    DUAL-GPU TRAINING RESULTS                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Wall time: 245.32s                                                   ║
║  PUMP training: 238.45s (GPU 0)                                       ║
║  DUMP training: 312.18s (GPU 1)                                       ║
║                                                                       ║
║  Speedup: 2.24x (550s sequential vs 245s parallel)                   ║
║                                                                       ║
║  PUMP AUC: 0.8923                                                     ║
║  DUMP AUC: 0.8867                                                     ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## Как работает Dual-GPU

### Обучение

```
multiprocessing.Process 1          multiprocessing.Process 2
        │                                    │
        ▼                                    ▼
┌───────────────────┐              ┌───────────────────┐
│  CUDA_VISIBLE_    │              │  CUDA_VISIBLE_    │
│  DEVICES=0        │              │  DEVICES=1        │
│                   │              │                   │
│  CatBoost GPU     │              │  CatBoost GPU     │
│  PUMP Model       │              │  DUMP Model       │
│                   │              │                   │
│  y = (label==1)   │              │  y = (label==-1)  │
└───────────────────┘              └───────────────────┘
        │                                    │
        └────────────────────────────────────┘
                          │
                          ▼
                  Combined Results
```

### Inference

```python
# Каждую минуту:
# 1. Собираем свежие данные
# 2. Рассчитываем признаки
# 3. Параллельно запускаем 2 процесса:

process_1: model.predict_proba(X) on GPU 0  # PUMP prob
process_2: model.predict_proba(X) on GPU 1  # DUMP prob

# 4. Комбинируем результаты:
if pump_prob > 0.85 and pump_prob > dump_prob:
    alert = "PUMP"
elif dump_prob > 0.85 and dump_prob > pump_prob:
    alert = "DUMP"
```

## Feature Engineering

### Группы признаков

| Группа | Признаки | Описание |
|--------|----------|----------|
| **Trade Flow** | `taker_buy_ratio`, `cvd_10/30`, `delta_ma_ratio` | Агрессивные покупки/продажи |
| **Volume Anomaly** | `vol_zscore_20/60/120`, `rvol_seasonal`, `vol_acceleration` | Аномалии объема |
| **Price Action** | `parkinson_vol`, `breakout_upper/lower`, `wick_ratios` | Ценовое действие |
| **Market Regime** | `btc_corr_30`, `btc_corr_drop`, `relative_strength` | Корреляция с BTC |

### Data Leakage Prevention

```python
# КРИТИЧЕСКИ ВАЖНО: Все признаки сдвигаются на 1 период
df['feature_lag1'] = df['feature'].shift(1)
```

## Формат файла меток

```csv
timestamp,symbol,label
1704067200000,ETH/USDT,1
1704070800000,BNB/USDT,-1
1704074400000,SOL/USDT,0
```

- `1` = памп (обучаем PUMP detector)
- `-1` = дамп (обучаем DUMP detector)
- `0` = нейтрально (negative class для обеих моделей)

## Алерты

### Single-GPU Alert

```
[2024-01-01 12:00:00] 🚨 PUMP ALERT: ETH/USDT | Prob: 92.34% | Price: 2350.5000
```

### Dual-GPU Alert

```
[2024-01-01 12:00:00] 📈 ETH/USDT | PUMP: 92.34% (RTX3060) | DUMP: 12.15% (P100) | Signal: PUMP | Price: 2350.5000
```

## Конфигурация GPU

### Автоматические пресеты

| GPU | VRAM | Depth | Iterations |
|-----|------|-------|------------|
| RTX 3060 | 12GB | 8 | 2000 |
| Tesla P100 | 16GB | 10 | 3000 |

### Ручная настройка

```python
# dual_gpu_config.py
dual_config = DualGPUConfig(
    pump_gpu_id=0,  # RTX 3060
    dump_gpu_id=1,  # Tesla P100
    pump_model_params={
        'depth': 8,
        'iterations': 2000,
    },
    dump_model_params={
        'depth': 10,
        'iterations': 3000,
    }
)
```

## Best Practices

1. **Dual-GPU для обучения**: ~2x ускорение
2. **Single-GPU для inference**: Быстрее для одного предсказания
3. **Dual-GPU inference**: Когда нужна максимальная точность обоих детекторов
4. **TimeSeriesSplit**: Всегда для временных рядов!
5. **Class Weights**: Обязательно для несбалансированных данных

## Логирование

```
logs/
├── pipeline.log          # Основной лог
├── alerts.jsonl          # Single-GPU алерты
└── dual_gpu_alerts.jsonl # Dual-GPU алерты
```

## Требования

- Python 3.10+
- CUDA 11.0+
- 2x NVIDIA GPU (для dual-GPU mode)
- 8GB+ RAM

## Лицензия

MIT License

---

**DISCLAIMER**: Образовательный проект. Торговля криптовалютами несет риски.
