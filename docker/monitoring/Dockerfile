FROM python:3.10-slim

WORKDIR /app

COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY src/ml_pipeline/monitoring/ ./monitoring/
COPY configs/ ./configs/

CMD ["python", "-m", "monitoring.drift_detection"]