import onnx
import onnxruntime as ort
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
# Загрузка обученной модели
model = joblib.load('models/trained/credit_default_model.pkl')
#Предположим, что модель была обучена на 20 features
initial_type = [('float_input', FloatTensorType([None, 20]))]
onx = convert_sklearn(model, initial_types=initial_type)
# Сохранение ONNX модели
with open("models/trained/model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
# Проверка инференса
sess = ort.InferenceSession("models/trained/model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name