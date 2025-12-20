#!/bin/bash
set -e

ENVIRONMENT=${1:-staging}

cd "$(dirname "$0")/../environments/${ENVIRONMENT}"

# Получение kubeconfig
terraform output -raw kubeconfig > kubeconfig.yaml
export KUBECONFIG=$(pwd)/kubeconfig.yaml

# Проверка подключения
kubectl cluster-info

# Установка необходимых компонентов
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# Ожидание установки ingress-nginx
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s

echo "Kubernetes cluster for ${ENVIRONMENT} is ready!"