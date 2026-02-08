import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "CLIP-DLUT Backend"
    API_V1_STR: str = "/api/v1"
    
    # Storage
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "storage", "uploads")
    RESULT_DIR: str = os.path.join(os.getcwd(), "storage", "results")
    
    # Celery & Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Model Config
    DEVICE: str = "cuda" # or cpu, autodetect handled in code but good to have env var
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)
