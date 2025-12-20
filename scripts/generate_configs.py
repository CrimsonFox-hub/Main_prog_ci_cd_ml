# scripts/generate_configs.py
import yaml
import os
from jinja2 import Template

def generate_config(environment, template_path, output_path, variables):
    with open(template_path, 'r') as f:
        template = Template(f.read())
    
    rendered = template.render(**variables)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(rendered)

# Использование
generate_config(
    environment="production",
    template_path="configs/templates/api.yaml.j2",
    output_path="configs/production/api.yaml",
    variables={
        "db_host": os.getenv("DB_HOST"),
        "redis_host": os.getenv("REDIS_HOST"),
        # ... другие переменные
    }
)