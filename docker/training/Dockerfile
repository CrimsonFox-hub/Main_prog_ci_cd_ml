# ============================================
# ТРЕНИРОВОЧНЫЙ ОБРАЗ
# ============================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu-training

FROM python:3.10-slim AS cpu-training

# Для GPU версии, раскомментируйте строку выше и закомментируйте строку ниже
# FROM gpu-training AS base

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
COPY requirements.txt requirements-ml.txt requirements-dev.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-ml.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Установка PyTorch (CPU или GPU версия)
ARG TORCH_VERSION=2.1.0
ARG CUDA_VERSION=cpu  # Для GPU: cu118

RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+${CUDA_VERSION} \
    torchvision==0.16.0+${CUDA_VERSION} \
    torchaudio==2.1.0+${CUDA_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Установка ONNX и оптимизаторов
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime==1.16.3 \
    onnxruntime-gpu==1.16.3 \
    onnxoptimizer==0.3.13

# Установка DVC
RUN pip install --no-cache-dir dvc[s3] dvc-gdrive

# Копирование кода
COPY . .

# Создание директорий
RUN mkdir -p /app/data /app/models /app/logs /app/reports /app/artifacts

# Установка переменных окружения
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Команда по умолчанию
CMD ["python", "src/ml_pipeline/training/train_model.py"]