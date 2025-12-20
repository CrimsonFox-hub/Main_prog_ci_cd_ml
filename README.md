# Проект: Промышленное развертывание кредитной скоринговой системы с полным MLOps-циклом

## Описание
Данный проект представляет собой комплексное решение для автоматизации жизненного цикла модели машинного обучения для кредитного скоринга. Реализован полный MLOps-цикл: от подготовки модели и контейнеризации до развертывания в облачной инфраструктуре, настройки CI/CD, мониторинга и автоматического переобучения.

## Ключевые технологии
- **Модель и данные**: PyTorch, ONNX, Scikit-learn, DVC
- **Контейнеризация и оркестрация**: Docker, Docker Compose, Kubernetes
- **Инфраструктура как код**: Terraform (для Yandex Cloud / VK Cloud)
- **CI/CD**: GitHub Actions
- **Мониторинг и логирование**: Prometheus, Grafana, Loki
- **Мониторинг ML-моделей**: Evidently AI
- **Оркестрация пайплайнов**: Apache Airflow

## Структура проекта

``` bash
Main_prog_ci_cd_ml/
├── configs/ # Конфигурационные файлы
├── data/ # Данные и DVC-конфигурация
├── docker/ # Dockerfile для сервисов
├── infrastructure/ # Terraform-конфигурации
├── kubernetes/ # Манифесты для Kubernetes
├── models/ # Сохраненные модели и артефакты
├── notebooks/ # Jupyter-ноутбуки для анализа
├── reports/ # Отчеты по тестированию и бенчмаркам
├── scripts/ # Вспомогательные скрипты
├── src/ # Исходный код
│ ├── api/ # FastAPI-приложение
│ ├── ml_pipeline/ # Пайплайн обучения, инференса, мониторинга
│ └── utils/ # Вспомогательные утилиты
├── tests/ # Модульные и интеграционные тесты
└── dags/ # Airflow DAG для переобучения

```
## Быстрый старт (локально)

1. **Клонирование и установка**
    ```bash
    git clone https://github.com/CrimsonFox-hub/Main_prog_ci_cd_ml.git
    cd Main_prog_ci_cd_ml
    python -m venv venv
    # Для Windows: .\venv\Scripts\activate
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

2. **Загрузка данных и обучение модели**
    ```bash
    make data-download
    make data-process
    make train  # Обучает нейронную сеть и экспортирует в ONNX
    ```

3. **Запуск локальных сервисов**
    ```bash
    docker-compose up -d postgres mlflow minio
    make up  # Запуск API и инфраструктуры
    ```
    - API будет доступен на `http://localhost:8000/docs`
    - MLflow UI: `http://localhost:5000`

## Развертывание в облаке
Инструкции по развертыванию в Yandex Cloud с использованием Terraform и Kubernetes находятся в директории `infrastructure/README.md`.

## Мониторинг
После развертывания в облаке настроены дашборды Grafana для мониторинга:
- Метрики инфраструктуры (CPU, память, сеть)
- Метрики производительности модели (латентность, ошибки)
- Детекция дрифта данных и концепта (Evidently AI)
