FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt requirements-dev.txt ./

# Установка Python зависимостей
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Копирование кода
COPY . .

# Создание директорий
RUN mkdir -p /app/data /app/models /app/logs /app/reports

WORKDIR /app

CMD ["python", "src/ml_pipeline/training/train_model.py"]