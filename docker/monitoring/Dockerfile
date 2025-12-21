# ============================================
# МОНИТОРИНГ И ДРИФТ-ДЕТЕКШН
# ============================================

FROM python:3.11

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка зависимостей
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir evidently prometheus-client

# Копирование кода мониторинга
COPY src/ml_pipeline/monitoring/ ./monitoring/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Создание пользователя
RUN useradd -m -u 1000 monitor && \
    chown -R monitor:monitor /app

USER monitor

EXPOSE 8080

# Запуск сервиса мониторинга
CMD ["python", "-m", "monitoring.drift_monitor_service"]