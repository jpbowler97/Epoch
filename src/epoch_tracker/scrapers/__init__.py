"""Data scrapers for various sources."""

import json
from pathlib import Path
from typing import Union, Dict, Any

from .base import BaseScraper, ScraperError
from .web_scraper_base import WebScraperBase
from .file_scraper_base import FileScraperBase


def create_scraper(config: Union[str, Path, Dict[str, Any]]) -> Union[WebScraperBase, FileScraperBase]:
    """Create a appropriate scraper instance from configuration.
    
    Args:
        config: Configuration file path or dictionary
        
    Returns:
        WebScraperBase or FileScraperBase instance
    """
    # Load configuration if path provided
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise ScraperError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Pass the path to the scraper for direct loading
        config_to_pass = config_path
    else:
        config_dict = config
        config_to_pass = config
    
    # Validate required config fields
    required_fields = ['name', 'type', 'source', 'parser']
    for field in required_fields:
        if field not in config_dict:
            raise ScraperError(f"Configuration missing required field: {field}")
    
    # Determine scraper type based on source type
    source_type = config_dict['source'].get('type')
    name = config_dict['name']
    
    if source_type == 'http':
        return WebScraperBase(name, config_to_pass)
    elif source_type in ['local_files', 'local_file', 'directory']:
        return FileScraperBase(name, config_to_pass)
    else:
        raise ScraperError(f"Unsupported source type: {source_type}")


__all__ = [
    "BaseScraper",
    "create_scraper", 
    "WebScraperBase",
    "FileScraperBase"
]