@echo off
chcp 65001 >nul
echo ========================================
echo MLOps Credit Scoring - Windows Commands
echo ========================================
echo.

if "%1"=="init" goto init
if "%1"=="install" goto install
if "%1"=="train" goto train
if "%1"=="test" goto test
if "%1"=="up" goto up
if "%1"=="down" goto down
if "%1"=="clean" goto clean
if "%1"=="help" goto help
if "%1"=="" goto help

:help
echo Доступные команды:
echo   init     - Инициализация проекта
echo   install  - Установка зависимостей
echo   train    - Обучение модели
echo   test     - Запуск тестов
echo   up       - Запуск Docker сервисов
echo   down     - Остановка Docker сервисов
echo   clean    - Очистка проекта
echo   help     - Справка
goto end

:init
echo Инициализация проекта...
python -m venv venv
if not exist venv (
    echo Ошибка: Не удалось создать виртуальное окружение
    echo Убедитесь, что Python установлен и добавлен в PATH
    goto end
)
echo Виртуальное окружение создано
echo.
echo Для активации выполните:
echo   venv\Scripts\activate
echo.
echo Затем установите зависимости:
echo   pip install -r requirements.txt
echo   pip install -r requirements-dev.txt
goto end

:install
echo Установка зависимостей...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Ошибка: Не удалось активировать виртуальное окружение
    echo Сначала выполните: init
    goto end
)
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
echo Зависимости установлены
goto end

:train
echo Обучение модели...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Ошибка: Активируйте виртуальное окружение
    echo venv\Scripts\activate
    goto end
)
python src/ml_pipeline/training/train_model.py --config configs/training_config.yaml
echo Обучение завершено
goto end

:test
echo Запуск тестов...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Ошибка: Активируйте виртуальное окружение
    echo venv\Scripts\activate
    goto end
)
python -m pytest tests/unit/ -v
echo Тестирование завершено
goto end

:up
echo Запуск Docker сервисов...
docker-compose up -d
echo Сервисы запущены
echo.
echo Доступные сервисы:
echo   API:         http://localhost:8000
echo   MLflow:      http://localhost:5000
echo   MinIO:       http://localhost:9001
echo   Prometheus:  http://localhost:9090
echo   Grafana:     http://localhost:3000 (admin/admin)
goto end

:down
echo Остановка Docker сервисов...
docker-compose down
echo Сервисы остановлены
goto end

:clean
echo Очистка проекта...
if exist __pycache__ rmdir /s /q __pycache__
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .coverage del .coverage
if exist htmlcov rmdir /s /q htmlcov
if exist .mypy_cache rmdir /s /q .mypy_cache
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
echo Очистка завершена
goto end

:end
echo.
pause