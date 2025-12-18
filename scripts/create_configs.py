import os
import yaml

# Создаем директории
os.makedirs('configs', exist_ok=True)
os.makedirs('models/processed', exist_ok=True)
os.makedirs('models/trained', exist_ok=True)

# Processing config
processing_config = {
    'data': {
        'numerical_features': [
            'duration', 'credit_amount', 'age', 'installment_commitment',
            'residence_since', 'existing_credits', 'num_dependents'
        ],
        'categorical_features': [
            'checking_status', 'credit_history', 'purpose', 'savings_status',
            'employment', 'personal_status', 'other_parties', 'property_magnitude',
            'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
        ],
        'target_column': 'default'
    },
    'model': {
        'paths': {
            'scaler': 'models/processed/scaler.pkl',
            'encoder': 'models/processed/encoder.pkl'
        }
    }
}

# Training config
training_config = {
    'model_paths': {
        'trained': 'models/trained/model.pkl',
        'onnx': 'models/trained/model.onnx',
        'tensorflow': 'models/trained/model'
    },
    'data': {
        'train_path': 'data/processed/train.csv',
        'test_path': 'data/processed/test.csv',
        'target_column': 'default'
    }
}

# Сохраняем конфиги
with open('configs/processing_config.yaml', 'w') as f:
    yaml.dump(processing_config, f, default_flow_style=False, allow_unicode=True)

with open('configs/training_config.yaml', 'w') as f:
    yaml.dump(training_config, f, default_flow_style=False, allow_unicode=True)

print("✅ Конфигурационные файлы созданы")