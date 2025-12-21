#!/bin/bash
# scripts/yandex/deploy-on-vm.sh
set -e

VM_IP=$1
if [ -z "$VM_IP" ]; then
    echo "Usage: $0 <VM_PUBLIC_IP>"
    exit 1
fi

echo "Copying project to VM..."
scp -r . ubuntu@$VM_IP:/home/ubuntu/mlops-credit-scoring

echo "Running docker-compose on VM..."
ssh ubuntu@$VM_IP "cd /home/ubuntu/mlops-credit-scoring && docker-compose -f docker-compose.yc.yml up -d"

echo "Services are starting on VM..."
echo "API: http://$VM_IP:8000"
echo "MLflow: http://$VM_IP:5000"
echo "Grafana: http://$VM_IP:3000"
echo "MinIO Console: http://$VM_IP:9001"