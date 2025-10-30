from telnetlib import SE
from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv
from sqlalchemy.engine import URL

userdb =  os.environ.get('USER_DATABASE') if os.environ.get('USER_DATABASE') else 'postgres'
passworddb = os.environ.get('USER_DATABASE') if os.environ.get('USER_DATABASE') else 'postgres'

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL') if os.getenv('DATABASE_URL') else ''
SECRET_KEY = os.getenv('SECRET_KEY') if os.getenv('SECRET_KEY') else ''

class Settings(BaseSettings):
    PROJECT_NAME: str = "Hospital Agent"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = DATABASE_URL
    
    # Security
    SECRET_KEY: str = SECRET_KEY if SECRET_KEY else ''
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    