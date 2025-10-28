# Credit Scoring Model Pipeline

Учебный проект по дисциплине «Автоматизация процессов разработки и тестирования моделей машинного обучения». Репозиторий содержит полнофункциональный ML-пайплайн для PD-модели (кредитный скоринг), покрывающий этапы от подготовки данных до мониторинга и развертывания.

## Содержание
- [Структура проекта](#структура-проекта)
- [Установка и подготовка окружения](#установка-и-подготовка-окружения)
- [Работа с данными и DVC](#работа-с-данными-и-dvc)
- [Обучение и эксперименты](#обучение-и-эксперименты)
- [Валидация данных](#валидация-данных)
- [API и Docker](#api-и-docker)
- [Мониторинг дрифта](#мониторинг-дрифта)
- [CI/CD](#cicd)

## Структура проекта
```
├── data/                  # Исходные и подготовленные данные (под управлением DVC)
├── models/                # Сохранённые модели и артефакты обучения
├── src/
│   ├── data/              # Скрипты подготовки данных и проверки качества
│   ├── models/            # Pipeline, обучение, запуск экспериментов
│   ├── api/               # FastAPI приложение и утилиты инференса
│   └── monitoring/        # Скрипт расчёта PSI и проверки дрифта
├── tests/                 # Unit-тесты и примеры валидных датасетов
├── dvc.yaml               # DVC-пайплайн prepare → validate → quality → train
├── Dockerfile             # Образ с API и всеми зависимостями
└── README.md
```

## Установка и подготовка окружения
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
# Для работы с DVC установите отдельно: pip install dvc
```

### Дополнительно
- Для работы с DVC установите [Git LFS](https://git-lfs.com/), если требуется хранить большие файлы.
- Для локального трекинга экспериментов MLflow используется каталог `mlruns/`.

## Работа с данными и DVC
1. Получите исходный датасет `UCI_Credit_Card.csv` из репозитория UCI и положите в `data/raw/` (или скачайте командой `dvc pull`, если настроено удалённое хранилище).
2. Запустите пайплайн DVC:
   ```bash
   dvc repro
   ```
   Пайплайн включает стадии:
   - `prepare` — очистка и разделение данных (`src/data/make_dataset.py`).
   - `validate` — проверка данных Pandera (`src/data/validation.py`).
   - `quality` — генерация отчёта и дополнительных проверок (`src/data/quality_report.py`).
   - `train` — обучение модели и логирование метрик в MLflow (`src/models/train.py`).

## Обучение и эксперименты
- Основной скрипт обучения: `python -m src.models.train`.
- Для серии экспериментов с различными моделями запустите:
  ```bash
  python -m src.models.experiments --data_dir data/processed --mlruns_dir mlruns
  ```
  Скрипт выполнит не менее пяти прогонов с разными алгоритмами и гиперпараметрами и сохранит результаты в MLflow UI (запуск UI: `mlflow ui --backend-store-uri mlruns`).

## Валидация данных
- Формальные правила описаны в `src/data/validation.py` (Pandera).
- Тесты на схему находятся в `tests/test_data_validation.py`.
- Быструю проверку можно выполнить командой:
  ```bash
  python -m src.data.validation --train data/processed/train.csv --test data/processed/test.csv --out data/processed/validation_report.json
  ```

## API и Docker
- FastAPI-приложение располагается в `src/api/app.py` и предоставляет endpoint `POST /predict`.
- Для запуска API локально:
  ```bash
  uvicorn src.api.app:app --reload --port 8000
  ```
- Docker:
  ```bash
  docker build -t credit-default-api .
  docker run -p 8000:8000 -e CREDIT_MODEL_PATH=models/credit_default_model.pkl credit-default-api
  ```
  После запуска API доступно по адресу `http://localhost:8000/docs`.

## Мониторинг дрифта
Скрипт `src/monitoring/psi_monitor.py` имитирует поступление новых данных: берёт батч примеров, опционально отправляет их на API для получения вероятностей и рассчитывает PSI. Пример запуска:
```bash
python -m src.monitoring.psi_monitor \
  --train data/processed/train.csv \
  --new data/processed/test.csv \
  --model models/credit_default_model.pkl \
  --out reports/psi_report.json
```
При наличии запущенного API можно добавить `--api-url http://localhost:8000/predict`.

## CI/CD
- Workflow GitHub Actions (`.github/workflows/ci.yml`) запускает `black`, `flake8`, `pytest` и проверку данных Pandera на примере тестовых датасетов.
- Для поддержания качества кода рекомендуется перед коммитом выполнять:
  ```bash
  black src tests
  flake8 src tests
  pytest
  ```

## Полезные ссылки
- [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [Pandera Documentation](https://pandera.readthedocs.io/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
