#!/usr/bin/env python3
"""
Registry for managing both legacy and configurable scrapers.

This module provides a unified interface for loading and executing
both traditional hardcoded scrapers and new JSON-configured scrapers.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from epoch_tracker.scrapers import (
    BaseScraper,
    create_scraper
)
from epoch_tracker.models import Model, ModelCollection


class ScraperRegistry:
    """Registry for managing and executing scrapers."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the scraper registry.
        
        Args:
            config_dir: Directory containing JSON scraper configurations
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = config_dir or Path("configs/scrapers")
        self.scrapers = {}
        self._load_scrapers()
    
    def _load_scrapers(self):
        """Load all available scrapers (both legacy and configurable)."""
        # Load configurable scrapers from JSON files
        if self.config_dir.exists():
            for config_file in self.config_dir.glob("*.json"):
                try:
                    scraper_name = config_file.stem
                    self.scrapers[scraper_name] = {
                        'type': 'configurable',
                        'config_path': config_file,
                        'enabled': True
                    }
                    self.logger.info(f"Registered configurable scraper: {scraper_name}")
                except Exception as e:
                    self.logger.error(f"Failed to register scraper from {config_file}: {e}")
        
        # No more legacy scrapers - all are now configurable!
    
    def get_scraper(self, name: str) -> Optional[BaseScraper]:
        """
        Get a scraper instance by name.
        
        Args:
            name: Name of the scraper
            
        Returns:
            Scraper instance or None if not found
        """
        if name not in self.scrapers:
            self.logger.error(f"Scraper not found: {name}")
            return None
        
        scraper_info = self.scrapers[name]
        
        if not scraper_info['enabled']:
            self.logger.info(f"Scraper {name} is disabled")
            return None
        
        try:
            if scraper_info['type'] == 'configurable':
                # Create configurable scraper from JSON
                return create_scraper(scraper_info['config_path'])
            else:
                # Instantiate legacy scraper
                return scraper_info['class']()
        except Exception as e:
            self.logger.error(f"Failed to create scraper {name}: {e}")
            return None
    
    def list_scrapers(self, enabled_only: bool = False) -> List[str]:
        """
        List all available scrapers.
        
        Args:
            enabled_only: If True, only return enabled scrapers
            
        Returns:
            List of scraper names
        """
        if enabled_only:
            return [name for name, info in self.scrapers.items() if info['enabled']]
        return list(self.scrapers.keys())
    
    def run_scraper(self, name: str) -> Optional[ModelCollection]:
        """
        Run a single scraper by name.
        
        Args:
            name: Name of the scraper to run
            
        Returns:
            ModelCollection or None if scraper failed
        """
        scraper = self.get_scraper(name)
        if not scraper:
            return None
        
        try:
            self.logger.info(f"Running scraper: {name}")
            
            # Handle different scraper interfaces
            if hasattr(scraper, 'scrape'):
                # New configurable scraper interface
                models = scraper.scrape()
                return ModelCollection(models=models, source=name)
            else:
                # Legacy scraper interface
                return scraper.scrape_models()
                
        except Exception as e:
            self.logger.error(f"Scraper {name} failed: {e}")
            return None
    
    def run_all_scrapers(self, enabled_only: bool = True) -> Dict[str, ModelCollection]:
        """
        Run all scrapers and collect results.
        
        Args:
            enabled_only: If True, only run enabled scrapers
            
        Returns:
            Dictionary mapping scraper names to their ModelCollections
        """
        results = {}
        scrapers_to_run = self.list_scrapers(enabled_only=enabled_only)
        
        for scraper_name in scrapers_to_run:
            collection = self.run_scraper(scraper_name)
            if collection:
                results[scraper_name] = collection
                self.logger.info(f"Scraper {scraper_name} collected {len(collection.models)} models")
        
        return results
    
    def enable_scraper(self, name: str):
        """Enable a scraper."""
        if name in self.scrapers:
            self.scrapers[name]['enabled'] = True
            self.logger.info(f"Enabled scraper: {name}")
    
    def disable_scraper(self, name: str):
        """Disable a scraper."""
        if name in self.scrapers:
            self.scrapers[name]['enabled'] = False
            self.logger.info(f"Disabled scraper: {name}")
    
    def get_scraper_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a scraper."""
        return self.scrapers.get(name)


def create_registry(config_dir: Optional[Path] = None) -> ScraperRegistry:
    """
    Create a scraper registry instance.
    
    Args:
        config_dir: Optional directory for scraper configurations
        
    Returns:
        ScraperRegistry instance
    """
    return ScraperRegistry(config_dir)