# ============================================
# МОНИТОРИНГ И ДРИФТ-ДЕТЕКШН
# ============================================

FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements-ml.txt requirements.txt ./

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements-ml.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir evidently==0.4.14 prometheus-client==0.19.0

# Копирование кода мониторинга
COPY src/ml_pipeline/monitoring/ ./monitoring/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Создание пользователя
RUN useradd -m -u 1000 monitor && \
    chown -R monitor:monitor /app

USER monitor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Порт для метрик Prometheus
EXPOSE 8080

# Запуск сервиса мониторинга
CMD ["python", "-m", "monitoring.drift_monitor_service", \
     "--config", "/app/configs/monitoring.yaml", \
     "--port", "8080"]