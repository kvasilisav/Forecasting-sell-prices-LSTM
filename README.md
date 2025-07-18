# Time Series Forecasting with LSTM

## Описание проекта

Проект реализует прогнозирование временных рядов с использованием LSTM-сетей.
Модель обучается на последовательностях длиной 28 дней для предсказания цены на
следующий день.

### Особенности обработки данных

Данные преобразуются в скользящие окна:

- **Входные данные**: последовательность длиной 28
- **Целевое значение**: цена на следующий день

Пример преобразования:

```python
    def _create_sequences(self, data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length - 1):
            _x = data[i : (i + seq_length), :]
            _y = data[i + seq_length, 0]
            x.append(_x)
            y.append(_y)
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y).reshape(-1, 1))
```

Разделение данных:

80% данных - тренировочная выборка

20% данных - тестовая выборка

## Setup

### Настройка окуржения с помощью poetry

```bash
poetry env activate
poetry install
```

### Установка pre-commit и проверка файлов

```bash
pre-commit install
pre-commit run -a
```

### Поднятие mlflow локально

```bash
mlflow server --host 127.0.0.1 --port 8080
```

## Train

### Запуск обучения

```bash
python forecasting-m5/train.py
```

- при запуске автоматически подтягиваются данные с помощью:

  ```bash
  dvc pull
  ```

- препроцессинг:
  - нормализация
  - добавления фичей
  - создание последовательностей

## Production preparation

### ONNX

По окончании обучения в папку `model/checkpoints` сохраняются веса модели в
форматах `.ckpt` и `.onnx`. Эти же веса загружаются из DVC на случай, если
обучение модели не будет завершено.

### TensorRT

Для конвертации весов модели из формата ONNX в TensorRT можно использовать
скрипт:

```bash
bash scripts/totensorrt.sh
```

## Infer

```bash
python forecasting-m5/infer.py
```

Прогнозирование выполняется на всех данных, предварительно преобразованных в
последовательности цен за 28 дней. Результат сохраняется в файл
`predictions.csv`.
