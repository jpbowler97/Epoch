"""Configuration management for Epoch Tracker."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API tokens
    huggingface_token: Optional[str] = Field(None, env="HUGGINGFACE_TOKEN")
    
    # Scraping configuration
    default_delay: float = Field(1.0, env="DEFAULT_DELAY")
    huggingface_delay: float = Field(0.5, env="HUGGINGFACE_DELAY")
    max_requests_per_minute: int = Field(60, env="MAX_REQUESTS_PER_MINUTE")
    
    # Data storage
    data_dir: Path = Field(Path("./data"), env="DATA_DIR")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # FLOP formatting
    flop_format: str = Field("scientific", env="FLOP_FORMAT")  # "scientific" or "numeric"
    
    # HTTP settings
    timeout: int = Field(30, env="HTTP_TIMEOUT")
    user_agent: str = Field("Epoch AI Model Tracker 0.1.0", env="USER_AGENT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


_settings: Optional[Settings] = None
_config_data: Optional[Dict[str, Any]] = None


def load_config(config_file: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    global _config_data
    
    if _config_data is None:
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                _config_data = yaml.safe_load(f)
        else:
            _config_data = {}
    
    return _config_data


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    
    if _settings is None:
        _settings = Settings()
    
    return _settings


def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value by dot-separated key path.
    
    Args:
        key_path: Dot-separated path like 'scraping.default_delay'
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config = load_config()
    
    # Navigate through nested dict using key path
    value = config
    for key in key_path.split('.'):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value