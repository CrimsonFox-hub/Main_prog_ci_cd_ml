# ============================================
# БИЛДЕР: Установка зависимостей
# ============================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка зависимостей
COPY requirements.txt requirements-ml.txt ./

# Установка Python зависимостей в виртуальное окружение
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

# ============================================
# DVC СТЕЙДЖ: Загрузка данных и моделей
# ============================================
FROM python:3.10-slim AS dvc

WORKDIR /app

# Установка системных зависимостей для DVC
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей из builder
COPY --from=builder /opt/venv /opt/venv

# Установка DVC с поддержкой S3
RUN /opt/venv/bin/pip install --no-cache-dir dvc[s3]

# Копирование всего проекта (включая .dvc)
COPY . .

# Настройка DVC
RUN /opt/venv/bin/dvc remote add -d minio s3://credit-scoring-models \
    --force && \
    /opt/venv/bin/dvc remote modify minio endpointurl http://minio:9000

# Загрузка данных и моделей
RUN /opt/venv/bin/dvc pull -R models/ data/ || echo "DVC pull failed, continuing..."

# ============================================
# DEVELOPMENT СТЕЙДЖ (для локальной разработки)
# ============================================
FROM python:3.10-slim AS development

WORKDIR /app

# Копирование зависимостей из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Копирование исходного кода
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Копирование данных из DVC стейджа
COPY --from=dvc /app/models/ ./models/
COPY --from=dvc /app/data/ ./data/

# Создание пользователя
RUN useradd -m -u 1000 mlops && \
    chown -R mlops:mlops /app

USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV ENVIRONMENT=development

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================
# PRODUCTION СТЕЙДЖ (для продакшена)
# ============================================
FROM python:3.10-slim AS production

WORKDIR /app

# Копирование зависимостей из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка минимальных системных зависимостей
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование только необходимого кода
COPY src/ ./src/
COPY configs/production/ ./configs/

# Копирование моделей из DVC стейджа
COPY --from=dvc /app/models/ ./models/

# Создание пользователя
RUN useradd -m -u 1000 mlops && \
    chown -R mlops:mlops /app

USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV ENVIRONMENT=production
ENV PORT=8000

EXPOSE 8000

CMD ["gunicorn", "src.api.app:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]