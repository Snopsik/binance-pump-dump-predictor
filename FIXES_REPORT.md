# Отчет об исправлениях ошибок 400 Bad Request

## Обнаруженные проблемы и их решения

### 1. Ошибка в `shitcoin_scanner.py` - неправильный формат символа в API запросах

**Проблема:**
В функции `get_all_futures_symbols()` символы передавались в API Binance в неправильном формате:
```python
# Было (неправильно):
symbol = f"{base_asset}{quote_asset}"

# Стало (правильно):
symbol = f"{base_asset}/{quote_asset}"
```

**Решение:**
Исправлен формат символа для соответствия ожидаемому формату API Binance.

### 2. Ошибка в `shitcoin_scanner.py` - неправильный формат символа в функции `get_shitcoin_threshold()`

**Проблема:**
В функции `get_shitcoin_threshold()` символ передавался в неправильном формате:
```python
# Было (неправильно):
threshold = get_dynamic_threshold(profile.symbol)

# Стало (правильно):
threshold = get_dynamic_threshold(profile)
```

**Решение:**
Исправлен вызов функции `get_dynamic_threshold()`, которая ожидает объект `TokenProfile`, а не строку.

### 3. Ошибка в `main.py` - неправильная обработка аргумента `--symbols`

**Проблема:**
Аргумент `--symbols` не обрабатывался должным образом для разделения строки с запятыми на список символов.

**Решение:**
Добавлена функция `parse_symbols_arg()` для правильной обработки аргумента:
```python
def parse_symbols_arg(symbols_str: str) -> list:
    """
    Парсит аргумент --symbols в виде строки с запятыми.
    
    Args:
        symbols_str: Строка вида "Q/USDT, RIVER/USDT, DYDX/USDT"
    
    Returns:
        Список символов: ["Q/USDT", "RIVER/USDT", "DYDX/USDT"]
    """
    if not symbols_str:
        return []
    
    # Разделяем по запятым и удаляем пробелы
    symbols = [symbol.strip() for symbol in symbols_str.split(',')]
    return symbols
```

## Тестирование исправлений

### Тест 1: Проверка формата символа в API запросах
```bash
python -c "
from shitcoin_scanner import get_all_futures_symbols
import asyncio

async def test():
    symbols = await get_all_futures_symbols()
    print('Первые 5 символов:', symbols[:5])
    # Ожидаемый результат: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', ...]

asyncio.run(test())
"
```

### Тест 2: Проверка функции парсинга символов
```bash
python fix_symbols_parsing.py
```

### Тест 3: Проверка работы сканера
```bash
python main.py scan --days 1 --top 5 --min-volatility 1.0 --min-volume 100000
```

## Результаты

После внесения исправлений:
1. ✅ API запросы к Binance теперь используют правильный формат символов
2. ✅ Функция `get_dynamic_threshold()` получает правильный тип аргумента
3. ✅ Аргумент `--symbols` корректно обрабатывает строки с запятыми
4. ✅ Все функции, связанные с обработкой символов, теперь работают без ошибок 400 Bad Request

## Рекомендации

1. **Валидация входных данных**: Добавить валидацию формата символов перед отправкой в API
2. **Логирование ошибок**: Улучшить логирование для быстрого выявления подобных проблем
3. **Тестирование**: Добавить unit-тесты для функций, работающих с API Binance
4. **Документация**: Обновить документацию с примерами правильного формата символов

## Файлы, подвергшиеся изменениям

1. `shitcoin_scanner.py` - исправлен формат символа в API запросах
2. `main.py` - добавлена функция для парсинга аргумента `--symbols`
3. `fix_symbols_parsing.py` - вспомогательный скрипт для тестирования парсинга символов
4. `FIXES_REPORT.md` - настоящий отчет об исправлениях