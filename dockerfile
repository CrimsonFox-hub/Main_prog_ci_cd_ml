#Многостадийный Dockerfile с DVC

# Build stage
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
# DVC stage
FROM python:3.9-slim as dvc
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
RUN apt-get update && apt-get install -y git
RUN pip install dvc[s3]
COPY . .
RUN dvc pull -R
# Final stage
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=dvc /app/models ./models
COPY --from=dvc /app/data ./data
COPY api/ ./api
COPY app/ ./app
COPY config/ ./config
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]