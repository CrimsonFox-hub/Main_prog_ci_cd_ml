-- scripts/init-db.sql
CREATE DATABASE IF NOT EXISTS credit_scoring;
CREATE DATABASE IF NOT EXISTS mlflow;

CREATE USER IF NOT EXISTS mlflow_user WITH PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;