from pydantic import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Data Agent API"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # OpenAI Settings
    OPENAI_API_KEY: str
    OPENAI_PROJECT_ID: str
    
    # File Upload Settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {"csv", "xlsx", "xls"}
    
    # Security Settings
    EXECUTION_TIMEOUT: int = 30  # seconds
    MAX_MEMORY_USAGE: int = 512  # MB
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Create global settings object
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True) 