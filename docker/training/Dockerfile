# ============================================
# ТРЕНИРОВОЧНЫЙ ОБРАЗ
# ============================================

FROM python:3.11

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка зависимостей
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Установка PyTorch (CPU версия)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Установка ONNX
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime==1.16.3

# Копирование кода
COPY . .

# Создание директорий
RUN mkdir -p /app/data /app/models /app/logs /app/reports

# Установка переменных окружения
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Команда по умолчанию
CMD ["python", "src/ml_pipeline/training/train_model.py"]