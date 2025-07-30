"""
Configuration management for the Document Intelligence System.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    gemini_api_key: str = ""
    
    # Model Configuration
    model_name: str = "gemini-1.5-pro"
    embedding_model: str = "gemini-embedding-001"
    max_tokens: int = 4096
    temperature: float = 0.1
    
    # Database and Storage
    chroma_db_path: str = "./data/chroma_db"
    chroma_persist_directory: str = "./data/chroma_db"  # Added for compatibility
    documents_path: str = "./data/documents"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Processing Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    max_document_size: int = 10485760  # 10MB in bytes
    
    # Logging
    log_level: str = "INFO"
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def validate_settings() -> bool:
    """Validate that all required settings are present."""
    settings = get_settings()
    
    # For demo purposes, don't require Gemini API key
    # if not settings.gemini_api_key:
    #     raise ValueError("GEMINI_API_KEY is required but not set")
    
    # Create necessary directories
    os.makedirs(settings.chroma_db_path, exist_ok=True)
    os.makedirs(settings.documents_path, exist_ok=True)
    
    return True
