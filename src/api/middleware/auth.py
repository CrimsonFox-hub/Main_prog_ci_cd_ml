"""
Middleware для аутентификации и авторизации
"""
import jwt
import time
from typing import Optional, Dict, Any, List
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import wraps

from src.utils.logger import api_logger
from src.utils.config_loader import get_config

class JWTToken:
    """Класс для работы с JWT токенами"""
    
    def __init__(self):
        self.config = get_config('api')
        self.secret_key = self.config['jwt']['secret_key']
        self.algorithm = self.config['jwt']['algorithm']
        self.access_token_expire_minutes = self.config['jwt']['access_token_expire_minutes']
        self.refresh_token_expire_days = self.config['jwt']['refresh_token_expire_days']
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Создание access токена"""
        to_encode = data.copy()
        expire = time.time() + (self.access_token_expire_minutes * 60)
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.secret_key, 
            algorithm=self.algorithm
        )
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Создание refresh токена"""
        to_encode = data.copy()
        expire = time.time() + (self.refresh_token_expire_days * 24 * 60 * 60)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.secret_key, 
            algorithm=self.algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Верификация токена"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            api_logger.warning("Token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            api_logger.warning("Invalid token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Обновление access токена"""
        payload = self.verify_token(refresh_token)
        
        if payload.get('type') != 'refresh':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Создание нового access токена
        access_token_data = {
            "sub": payload.get("sub"),
            "user_id": payload.get("user_id"),
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", [])
        }
        
        return self.create_access_token(access_token_data)

class AuthMiddleware(HTTPBearer):
    """Middleware для аутентификации"""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.jwt_token = JWTToken()
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """Проверка аутентификации"""
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
        
        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
        
        # Верификация токена
        payload = self.jwt_token.verify_token(credentials.credentials)
        
        # Добавление пользователя в request state
        request.state.user = payload
        
        return payload

def require_auth():
    """Декоратор для проверки аутентификации"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request or not hasattr(request.state, 'user'):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(required_roles: List[str]):
    """Декоратор для проверки ролей"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request or not hasattr(request.state, 'user'):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user_roles = request.state.user.get('roles', [])
            
            # Проверка ролей
            if not any(role in user_roles for role in required_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_permission(required_permissions: List[str]):
    """Декоратор для проверки пермишенов"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request or not hasattr(request.state, 'user'):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user_permissions = request.state.user.get('permissions', [])
            
            # Проверка пермишенов
            if not all(perm in user_permissions for perm in required_permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

class APIKeyAuth:
    """Аутентификация по API ключу"""
    
    def __init__(self):
        self.config = get_config('api')
        self.api_keys = self.config.get('api_keys', {})
    
    async def __call__(self, request: Request):
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Добавление информации о клиенте
        request.state.client = self.api_keys[api_key]
        
        return request.state.client