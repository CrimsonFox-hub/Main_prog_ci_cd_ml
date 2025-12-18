# main_pipeline.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Загрузка данных
df = pd.read_csv("data/processed/train.csv")
print("Колонки:", df.columns.tolist())  # ← смотрим реальные названия

# 2. Разделение на признаки и целевую переменную
X = df.drop("target", axis=1)  # ← убедитесь, что имя правильное
y = df["target"]

# 3. Обучение модели
model = RandomForestClassifier()
model.fit(X, y)

# 4. Сохранение
joblib.dump(model, "models/trained/main_model.pkl")
print("Основная модель обучена!")