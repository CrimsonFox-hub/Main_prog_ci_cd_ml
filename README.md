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

# 1. Установите базовые системные утилиты и Docker с Docker Compose
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common git
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

# 2. Добавьте вашего пользователя в группу docker, чтобы не использовать sudo
```bash
sudo usermod -aG docker $USER

```
# Важно! Для применения изменения нужно выйти и зайти заново в SSH-сессию.
# Сделайте это после завершения всех команд этой части.

# 3. Клонируйте ваш проект с GitHub на ВМ
```bash
git clone https://github.com/CrimsonFox-hub/Main_prog_ci_cd_ml.git
cd Main_prog_ci_cd_ml

# 4. Настройте файл окружения (env). Создайте его на основе примера.
cp .env.example .env
# ОТКРОЙТЕ файл .env в редакторе (nano .env) и заполните реальными данными,
# например, паролями для БД. Для учебного проекта можно оставить упрощенные значения.
```
# установка docker-compose
```bash
apt install docker-compose
```
# 5. Запустите все сервисы
```bash
docker-compose up -d
```
# 6. Проверьте, что все контейнеры работают
```bash
docker-compose ps
```
# 7. Посмотрите логи (особенно если что-то не запустилось)
```bash
docker-compose logs api  # Или имя другого сервиса
```


## Мониторинг
После развертывания в облаке настроены дашборды Grafana для мониторинга:
- Метрики инфраструктуры (CPU, память, сеть)
- Метрики производительности модели (латентность, ошибки)
- Детекция дрифта данных и концепта (Evidently AI)



