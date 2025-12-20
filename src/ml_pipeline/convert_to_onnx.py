import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle

class CreditScoringNN(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=1):
        super(CreditScoringNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def convert_sklearn_to_onnx(model, input_size, output_path):
    """Convert scikit-learn model to ONNX via dummy neural network"""
    # Create a wrapper model
    class SklearnWrapper(nn.Module):
        def __init__(self, sklearn_model):
            super().__init__()
            self.sklearn_model = sklearn_model
            
        def forward(self, x):
            # Convert to numpy for sklearn prediction
            x_np = x.detach().numpy()
            y_pred = self.sklearn_model.predict_proba(x_np)[:, 1]
            return torch.from_numpy(y_pred).unsqueeze(1).float()
    
    wrapper = SklearnWrapper(model)
    
    # Export to ONNX
    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Validate the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    return output_path

def main():
    # Load your trained model
    model = joblib.load('models/credit_scoring_model.pkl')
    
    # Convert to ONNX
    input_size = len(model.feature_importances_) if hasattr(model, 'feature_importances_') else 20
    convert_sklearn_to_onnx(model, input_size, 'models/credit_scoring_model.onnx')
    
    print("Model converted to ONNX successfully")

if __name__ == "__main__":
    main()