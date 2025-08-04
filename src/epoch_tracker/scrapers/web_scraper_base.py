"""Base class for web-based data acquisition scrapers."""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .base import BaseScraper, ScraperNetworkError, ScraperError
from .parsers.web_parser import WebParser
from .utils.model_factory import ModelFactory
from ..models import Model, ModelCollection


class WebScraperBase(BaseScraper):
    """Base class for web-based scrapers that fetch data via HTTP."""
    
    def __init__(self, name: str, config: Optional[Union[Dict[str, Any], str, Path]] = None):
        """Initialize web scraper with configuration.
        
        Args:
            name: Scraper name
            config: Configuration dictionary, JSON file path, or None
        """
        # Load configuration from file if path provided
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise ScraperError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        config = config or {}
        
        # Convert JSON config to internal format if needed
        if self._is_json_config_format(config):
            config = self._convert_json_config(config)
        
        base_url = config.get('source', {}).get('url', '')
        
        super().__init__(name=name, base_url=base_url)
        
        self.config = config
        self.model_factory = ModelFactory()
        
        # Initialize web parser
        parser_config = config.get('source', {})
        self.web_parser = WebParser(parser_config)
        
        self.logger.info(f"Initialized web scraper: {name}")
    
    def _is_json_config_format(self, config: Dict[str, Any]) -> bool:
        """Check if config is in JSON format (vs internal format).
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if JSON format, False if internal format
        """
        # JSON format has 'type', 'source', 'parser' structure
        # Internal format has parser config directly in 'source'
        return ('type' in config and 
                'source' in config and 
                'parser' in config and
                config['source'].get('type') == 'http')
    
    def _convert_json_config(self, json_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON configuration to internal WebScraperBase format.
        
        Args:
            json_config: JSON configuration dictionary
            
        Returns:
            Internal configuration format
        """
        source = json_config['source']
        parser = json_config['parser']
        metadata = json_config.get('metadata', {})
        
        # Build web scraper config
        scraper_config = {
            'source': {
                'url': source['url'],
                'headers': source.get('headers', {}),
                'timeout': source.get('timeout', 30),
                'retries': source.get('retries', 3),
                'save_raw': source.get('save_raw', False),
                'cache_dir': source.get('cache_dir')
            }
        }
        
        # Add parser configuration to source
        if 'table_selector' in parser:
            scraper_config['source']['table_selector'] = parser['table_selector']
        
        if 'column_detection' in parser:
            scraper_config['column_mapping'] = parser['column_detection']
        elif 'columns' in parser:
            scraper_config['column_mapping'] = parser['columns']
        
        if 'benchmarks' in parser:
            scraper_config['benchmark_mapping'] = parser['benchmarks']
        
        # Add parser config for model creation
        scraper_config['parser'] = parser
        
        # Add model factory configuration
        scraper_config['metadata'] = {
            'source_name': metadata.get('source_name', json_config.get('name', '')),
            'source_url': metadata.get('source_url', source['url']),
            'developer_field': parser.get('developer_field'),
            'license_field': parser.get('license_field'),
            'description': metadata.get('description')
        }
        
        return scraper_config
    
    def scrape_models(self) -> ModelCollection:
        """Scrape models from web source.
        
        Returns:
            Collection of scraped models
        """
        source_config = self.config.get('source', {})
        parser_config = self.config.get('parser', {})
        
        url = source_config.get('url')
        if not url:
            raise ScraperNetworkError("No URL specified in source configuration")
        
        self.logger.info(f"Starting web scraping from: {url}")
        
        try:
            # Fetch and parse web data
            raw_rows = self.web_parser.fetch_and_parse(url, parser_config)
            
            if not raw_rows:
                self.logger.warning("No data extracted from web source")
                return ModelCollection(models=[], source=self.name)
            
            # Convert raw data to models
            models = self._create_models_from_data(raw_rows)
            
            # Save raw data for debugging/analysis
            self._save_raw_data(raw_rows, url)
            
            self.logger.info(f"Successfully scraped {len(models)} models from web source")
            
            return ModelCollection(models=models, source=self.name)
            
        except Exception as e:
            raise ScraperNetworkError(f"Failed to scrape web source {url}: {e}")
        finally:
            # Clean up web parser resources
            self.web_parser.close()
    
    def _create_models_from_data(self, raw_rows: List[Dict[str, Any]]) -> List[Model]:
        """Convert raw row data to Model objects.
        
        Args:
            raw_rows: List of raw data dictionaries
            
        Returns:
            List of Model objects
        """
        parser_config = self.config.get('parser', {})
        metadata_config = self.config.get('metadata', {})
        
        # The WebParser already mapped columns, so we need to create a column_config
        # that references the mapped field names directly
        column_detection = parser_config.get('column_detection', {})
        if column_detection:
            # Convert column_detection to direct field mapping
            column_config = {
                'columns': {k: k for k in column_detection.keys()},  # Direct field name mapping
                'benchmarks': parser_config.get('benchmarks', {}),
                'metadata': parser_config.get('metadata', {})
            }
        else:
            column_config = parser_config
        
        # Prepare source configuration for model factory
        # Don't pass source name for benchmark prefixing since benchmarks already include source
        source_config = {
            'name': '',  # Empty to avoid double-prefixing benchmarks
            'source_url': self.config.get('source', {}).get('url'),
            'description': metadata_config.get('description', f'{self.name} web scraper')
        }
        
        return self.model_factory.batch_create_models(raw_rows, column_config, source_config)
    
    def scrape(self) -> List[Model]:
        """Execute scraping and return list of models.
        
        Returns:
            List of scraped Model objects
        """
        collection = self.scrape_models()
        return collection.models if collection else []
    
    def to_collection(self) -> ModelCollection:
        """Get scraped data as a ModelCollection.
        
        Returns:
            ModelCollection with scraped models
        """
        return self.scrape_models()
    
    def _save_raw_data(self, raw_rows: List[Dict[str, Any]], source_url: str):
        """Save raw scraped data for debugging.
        
        Args:
            raw_rows: Raw data to save
            source_url: Source URL for reference
        """
        try:
            # Create data structure for saving
            save_data = {
                'source_url': source_url,
                'scraper_name': self.name,
                'row_count': len(raw_rows),
                'raw_data': raw_rows
            }
            
            # Save using base class method
            filename = f"web_{self.name}_latest"
            self.save_raw_data(save_data, filename)
            
        except Exception as e:
            self.logger.warning(f"Failed to save raw data: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        super().__exit__(exc_type, exc_val, exc_tb)
        if hasattr(self, 'web_parser'):
            self.web_parser.close()