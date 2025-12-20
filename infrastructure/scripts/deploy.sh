#!/bin/bash
set -e

ENVIRONMENT=${1:-staging}
ACTION=${2:-plan}

cd "$(dirname "$0")/../environments/${ENVIRONMENT}"

# Инициализация Terraform
terraform init \
  -backend-config="access_key=${YC_ACCESS_KEY}" \
  -backend-config="secret_key=${YC_SECRET_KEY}"

# Создание/обновление переменных
cat > terraform.tfvars <<EOF
yc_token = "${YC_TOKEN}"
cloud_id = "${YC_CLOUD_ID}"
folder_id = "${YC_FOLDER_ID}"
zone = "${YC_ZONE:-ru-central1-a}"
EOF

case $ACTION in
  plan)
    terraform plan -var-file="terraform.tfvars"
    ;;
  apply)
    terraform apply -var-file="terraform.tfvars" -auto-approve
    ;;
  destroy)
    read -p "Are you sure you want to destroy ${ENVIRONMENT}? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      terraform destroy -var-file="terraform.tfvars" -auto-approve
    fi
    ;;
  *)
    echo "Usage: $0 [staging|production] [plan|apply|destroy]"
    exit 1
    ;;
esac