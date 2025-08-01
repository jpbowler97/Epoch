"""Base scraper class with common functionality."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..models import Model, ModelCollection


class BaseScraper(ABC):
    """Abstract base class for all scrapers."""
    
    def __init__(self, name: str, base_url: str = ""):
        """Initialize the scraper.
        
        Args:
            name: Name of the scraper (e.g., 'huggingface')
            base_url: Base URL for the source
        """
        self.name = name
        self.base_url = base_url
        self.logger = logging.getLogger(f"scrapers.{name}")
        
        # Simple defaults without config system
        self.enabled = True
        self.delay = 1.0  # Default delay between requests
        
        self.logger.info(f"Initialized {name} scraper")
    
    @abstractmethod
    def scrape_models(self) -> ModelCollection:
        """Scrape models from the source.
        
        Returns:
            Collection of scraped models
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if this scraper is enabled."""
        return self.enabled
    
    def save_raw_data(self, data: Any, filename: str) -> Path:
        """Save raw scraped data to disk.
        
        Args:
            data: Data to save (will be JSON serialized)
            filename: Filename without extension
            
        Returns:
            Path to saved file
        """
        import json
        
        # Use hardcoded data directory
        raw_dir = Path("data") / "raw" / self.name
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = raw_dir / f"{filename}.json"
        
        # Add timestamp to data
        if isinstance(data, dict):
            data["_scraped_at"] = datetime.utcnow().isoformat()
        elif isinstance(data, list):
            data = {
                "data": data,
                "_scraped_at": datetime.utcnow().isoformat()
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Saved raw data to {filepath}")
        return filepath
    
    def load_raw_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load previously saved raw data.
        
        Args:
            filename: Filename without extension
            
        Returns:
            Loaded data or None if not found
        """
        import json
        
        # Use hardcoded data directory
        filepath = Path("data") / "raw" / self.name / f"{filename}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load raw data from {filepath}: {e}")
            return None
    
    def create_model(
        self,
        name: str,
        developer: str,
        **kwargs
    ) -> Model:
        """Create a model instance with common fields set.
        
        Args:
            name: Model name
            developer: Developer organization
            **kwargs: Additional model fields
            
        Returns:
            Model instance
        """
        # Set default source
        sources = kwargs.get('sources', [])
        if self.base_url and self.base_url not in sources:
            sources.append(f"{self.base_url} ({self.name} scraper)")
        
        return Model(
            name=name,
            developer=developer,
            sources=sources,
            **kwargs
        )
    
    def filter_models(self, models: List[Model], **criteria) -> List[Model]:
        """Filter models based on criteria.
        
        Args:
            models: List of models to filter
            **criteria: Filter criteria (e.g., min_parameters=1000000)
            
        Returns:
            Filtered list of models
        """
        filtered = models.copy()
        
        # Filter by minimum parameters
        if 'min_parameters' in criteria:
            min_params = criteria['min_parameters']
            filtered = [m for m in filtered if m.parameters and m.parameters >= min_params]
        
        # Filter by developer
        if 'developer' in criteria:
            dev = criteria['developer'].lower()
            filtered = [m for m in filtered if m.developer.lower() == dev]
        
        # Filter by release date
        if 'since_date' in criteria:
            since = criteria['since_date']
            filtered = [m for m in filtered if m.release_date and m.release_date >= since]
        
        return filtered
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        # Clean up HTTP session if it exists
        if hasattr(self, 'http') and hasattr(self.http, 'close'):
            self.http.close()


class ScraperError(Exception):
    """Base exception for scraper errors."""
    pass


class ScraperConfigError(ScraperError):
    """Configuration error in scraper."""
    pass


class ScraperNetworkError(ScraperError):
    """Network error during scraping."""
    pass