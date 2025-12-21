"""
Нейронная сеть для кредитного скоринга
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CreditScoringNN(nn.Module):
    """Нейронная сеть для кредитного скоринга"""
    
    def __init__(self, input_size, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        super(CreditScoringNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Создание слоев
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Выходной слой
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Прямой проход"""
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)
    
    def get_feature_importance(self):
        """Получение важности признаков"""
        # Используем веса первого слоя как меру важности
        if len(self.layers) > 0 and isinstance(self.layers[0], nn.Linear):
            weights = self.layers[0].weight.detach().cpu().numpy()
            importance = torch.mean(torch.abs(weights), dim=0)
            return importance.numpy()
        return None

class AdvancedCreditScoringNN(nn.Module):
    """Усовершенствованная нейронная сеть с attention механизмом"""
    
    def __init__(self, input_size, hidden_layers=[256, 128, 64], dropout_rate=0.3):
        super(AdvancedCreditScoringNN, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention слой
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """Прямой проход с attention"""
        # Энкодинг
        encoded = self.encoder(x)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(encoded), dim=1)
        
        # Weighted encoding
        weighted_encoding = encoded * attention_weights
        
        # Классификация
        output = self.classifier(weighted_encoding)
        return torch.sigmoid(output), attention_weights.squeeze()

def create_model(model_type='simple', input_size=20, **kwargs):
    """Фабрика для создания моделей"""
    if model_type == 'simple':
        return CreditScoringNN(input_size, **kwargs)
    elif model_type == 'advanced':
        return AdvancedCreditScoringNN(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Утилиты для модели
def save_model(model, path, metadata=None):
    """Сохранение модели с метаданными"""
    checkpoint = {
        'state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)

def load_model(path, model_class=CreditScoringNN, **model_args):
    """Загрузка модели с метаданными"""
    checkpoint = torch.load(path)
    model = model_class(**model_args)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint.get('metadata', {})