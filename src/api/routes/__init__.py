"""
Маршруты API
"""
from .health import router as health_router
from .predict import router as predict_router
from .models import router as models_router
from .monitoring import router as monitoring_router
from .admin import router as admin_router

__all__ = [
    'health_router',
    'predict_router',
    'models_router',
    'monitoring_router',
    'admin_router'
]