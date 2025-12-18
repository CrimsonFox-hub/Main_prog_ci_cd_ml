"""
Простой скрипт для запуска всего тестового пайплайна
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Выполнение команды с обработкой ошибок"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Команда: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ Успешно")
        if result.stdout:
            print(f"Вывод:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка (код {e.returncode}):")
        if e.stdout:
            print(f"Вывод:\n{e.stdout}")
        if e.stderr:
            print(f"Ошибка:\n{e.stderr}")
        return False

def main():
    print("🚀 Запуск упрощенного тестового пайплайна")
    print("="*60)
    
    # 1. Создание директорий
    print("\n1. Создание структуры директорий...")
    directories = [
        "data/raw",
        "data/processed",
        "models/trained",
        "logs",
        "reports",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Создана: {directory}")
    
    # 2. Создание тестовых данных
    print("\n2. Создание тестовых данных...")
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    np.random.seed(42)
    n_samples = 1000
    
    # Генерация данных
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_amount': np.random.randint(1000, 50000, n_samples),
        'loan_duration': np.random.randint(6, 60, n_samples),
        'payment_to_income': np.random.uniform(0.1, 0.5, n_samples),
        'existing_credits': np.random.randint(0, 5, n_samples),
        'dependents': np.random.randint(0, 5, n_samples),
        'residence_since': np.random.randint(0, 20, n_samples),
        'installment_rate': np.random.uniform(1.0, 4.0, n_samples),
        'target': np.random.binomial(1, 0.3, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Разделение на train/test
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['target']
    )
    
    # Сохранение
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"   Создано {len(train_df)} train и {len(test_df)} test образцов")
    print(f"   Дефолтов в train: {train_df['target'].mean():.2%}")
    print(f"   Дефолтов в test: {test_df['target'].mean():.2%}")
    
    # 3. Создание конфигурации
    print("\n3. Создание конфигурации...")
    config_content = """project:
  name: "credit-scoring-test"
  version: "1.0.0"

data:
  train_path: "data/processed/train.csv"
  target_column: "target"
  
model:
  name: "SimpleCreditNN"
  hidden_layers: [64, 32]
  dropout_rate: 0.3
  
  paths:
    best_model: "models/trained/best_model.pth"
    final_model: "models/trained/final_model.pth"
    scaler: "models/trained/scaler.pkl"
    metrics: "models/trained/training_metrics.json"

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
"""
    
    with open("configs/simple_test.yaml", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("   Конфигурация создана: configs/simple_test.yaml")
    
    # 4. Обучение модели
    print("\n4. Запуск обучения...")
    
    # Создаем упрощенный скрипт обучения
    train_script = """
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from sklearn.preprocessing import StandardScaler

# Простая модель
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Загрузка данных
print("Загрузка данных...")
train_df = pd.read_csv("data/processed/train.csv")

# Подготовка данных
X = train_df.drop(columns=['target']).values.astype(np.float32)
y = train_df['target'].values.astype(np.float32)

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Сохранение скейлера
Path("models/trained").mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, "models/trained/scaler.pkl")

# Конвертация в тензоры
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# Создание модели
model = SimpleModel(X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
print("Обучение...")
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    # Предсказания и точность
    with torch.no_grad():
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_tensor).float().mean()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

# Сохранение модели
torch.save(model.state_dict(), "models/trained/model.pth")
print("Модель сохранена: models/trained/model.pth")

# Тестирование
print("\\nТестирование...")
test_df = pd.read_csv("data/processed/test.csv")
X_test = test_df.drop(columns=['target']).values.astype(np.float32)
y_test = test_df['target'].values.astype(np.float32)
X_test_scaled = scaler.transform(X_test)

model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    predictions = model(X_test_tensor)
    predictions_bin = (predictions > 0.5).float()
    accuracy = (predictions_bin == torch.FloatTensor(y_test).unsqueeze(1)).float().mean()
    
print(f"Точность на тесте: {accuracy.item():.4f}")
print("✅ Обучение завершено успешно!")
"""
    
    with open("train_simple.py", "w", encoding="utf-8") as f:
        f.write(train_script)
    
    # Запуск обучения
    result = run_command("python train_simple.py", "Обучение модели")
    
    # 5. Очистка временных файлов
    if Path("train_simple.py").exists():
        Path("train_simple.py").unlink()
    
    print("\n" + "="*60)
    if result:
        print("🎉 Тестовый пайплайн успешно выполнен!")
        print("Созданы:")
        print("  - data/processed/train.csv")
        print("  - data/processed/test.csv")
        print("  - models/trained/model.pth")
        print("  - models/trained/scaler.pkl")
    else:
        print("⚠️ В процессе выполнения возникли ошибки")
    
    print("="*60)

if __name__ == "__main__":
    main()