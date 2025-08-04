"""Base class for file-based data acquisition scrapers."""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .base import BaseScraper, ScraperError
from .parsers.csv_parser import CSVParser
from .parsers.file_parser import FileParser
from .parsers.markdown_parser import MarkdownParser
from .utils.model_factory import ModelFactory
from ..models import Model, ModelCollection


class FileScraperBase(BaseScraper):
    """Base class for file-based scrapers that process local files."""
    
    def __init__(self, name: str, config: Optional[Union[Dict[str, Any], str, Path]] = None):
        """Initialize file scraper with configuration.
        
        Args:
            name: Scraper name
            config: Configuration dictionary, JSON file path, or None
        """
        super().__init__(name=name, base_url="")
        
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
        
        self.config = config
        self.model_factory = ModelFactory()
        
        # Initialize appropriate parser based on config
        self.parser = self._create_parser()
        
        self.logger.info(f"Initialized file scraper: {name}")
    
    def _is_json_config_format(self, config: Dict[str, Any]) -> bool:
        """Check if config is in JSON format (vs internal format).
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if JSON format, False if internal format
        """
        # JSON format has 'type', 'source', 'parser' structure
        # Internal format has direct 'type'/'file_type' and parser config
        return ('type' in config and 
                'source' in config and 
                'parser' in config and
                config['source'].get('type') in ['local_files', 'local_file', 'directory'])
    
    def _convert_json_config(self, json_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON configuration to internal FileScraperBase format.
        
        Args:
            json_config: JSON configuration dictionary
            
        Returns:
            Internal configuration format
        """
        source = json_config['source']
        parser = json_config['parser']
        metadata = json_config.get('metadata', {})
        file_type = json_config['type']
        
        # Build file scraper config
        scraper_config = {
            'type': self._map_file_type(file_type),
            'source': {}
        }
        
        # Handle different source specifications
        if 'paths' in source:
            scraper_config['source']['paths'] = source['paths']
        elif 'path' in source:
            scraper_config['source']['path'] = source['path']
        
        if 'fallback_pattern' in source:
            scraper_config['source']['fallback_pattern'] = source['fallback_pattern']
        
        if 'directory' in source:
            scraper_config['source']['directory'] = source['directory']
            scraper_config['source']['pattern'] = source.get('pattern', '*')
        
        # Add parser configuration
        scraper_config['parser'] = parser.copy()
        
        if 'delimiter' in parser:
            scraper_config['source']['delimiter'] = parser['delimiter']
        
        if 'encoding' in parser:
            scraper_config['source']['encoding'] = parser['encoding']
            
        # Add model factory configuration
        scraper_config['metadata'] = {
            'source_name': metadata.get('source_name', json_config.get('name', '')),
            'source_url': metadata.get('source_url', 'Local file'),
            'developer_field': parser.get('developer_field'),
            'license_field': parser.get('license_field'),
            'description': metadata.get('description')
        }
        
        return scraper_config
    
    def _map_file_type(self, config_type: str) -> str:
        """Map configuration file type to parser type.
        
        Args:
            config_type: Type from configuration
            
        Returns:
            Parser type string
        """
        type_mapping = {
            'csv_file': 'csv',
            'csv': 'csv',
            'html_file': 'html',
            'html': 'html',
            'markdown_file': 'markdown',
            'markdown': 'markdown',
            'md': 'markdown',
            'web_table': 'html',
            'local_html': 'html'
        }
        
        return type_mapping.get(config_type, config_type)
    
    def _create_parser(self):
        """Create appropriate parser based on configuration.
        
        Returns:
            Parser instance
        """
        # Try both 'type' and 'file_type' for compatibility
        source_type = self.config.get('type', self.config.get('file_type', ''))
        parser_config = self.config.get('source', {})
        
        # Map various type specifications to parsers
        if source_type in ['csv_file', 'csv']:
            return CSVParser(parser_config)
        elif source_type in ['html_file', 'html']:
            return FileParser(parser_config)
        elif source_type in ['markdown_file', 'markdown', 'md']:
            return MarkdownParser(parser_config)
        else:
            raise ScraperError(f"Unsupported file type: {source_type}")
    
    def scrape_models(self) -> ModelCollection:
        """Scrape models from local files.
        
        Returns:
            Collection of scraped models
        """
        source_config = self.config.get('source', {})
        parser_config = self.config.get('parser', {})
        # Try both 'type' and 'file_type' for compatibility
        source_type = self.config.get('type', self.config.get('file_type', ''))
        
        self.logger.info(f"Starting file scraping with type: {source_type}")
        
        try:
            # Parse files based on type
            if source_type in ['csv_file', 'csv']:
                raw_rows = self.parser.parse_files(source_config, parser_config)
            elif source_type in ['html_file', 'html', 'markdown_file', 'markdown', 'md']:
                # For single file types, get the file path
                file_path = self._get_file_path(source_config)
                raw_rows = self.parser.parse_file(file_path, parser_config)
            else:
                raise ScraperError(f"Unknown file type: {source_type}")
            
            if not raw_rows:
                self.logger.warning("No data extracted from files")
                return ModelCollection(models=[], source=self.name)
            
            # Convert raw data to models
            models = self._create_models_from_data(raw_rows)
            
            # Save raw data for debugging/analysis
            self._save_raw_data(raw_rows, source_type)
            
            self.logger.info(f"Successfully scraped {len(models)} models from files")
            
            return ModelCollection(models=models, source=self.name)
            
        except Exception as e:
            raise ScraperError(f"Failed to scrape files: {e}")
    
    def _get_file_path(self, source_config: Dict[str, Any]) -> str:
        """Get file path from source configuration.
        
        Args:
            source_config: Source configuration
            
        Returns:
            File path string
        """
        # Try different possible file path keys
        file_path = (source_config.get('file_path') or 
                    source_config.get('path') or 
                    source_config.get('paths', [None])[0])
        
        if not file_path:
            raise ScraperError("No file path specified in source configuration")
        
        return file_path
    
    def _create_models_from_data(self, raw_rows: List[Dict[str, Any]]) -> List[Model]:
        """Convert raw row data to Model objects.
        
        Args:
            raw_rows: List of raw data dictionaries
            
        Returns:
            List of Model objects
        """
        parser_config = self.config.get('parser', {})
        metadata_config = self.config.get('metadata', {})
        
        # The CSVParser already mapped columns, so we need to create a column_config
        # that references the mapped field names directly
        column_config = {
            'columns': {
                'model': 'model',  # Direct field name after mapping
                'developer': 'developer',
                'score': 'score',
                'rank': 'rank',
                'license': 'license'
            },
            'benchmarks': parser_config.get('benchmarks', {}),
            'metadata': parser_config.get('metadata', {})
        }
        
        # Prepare source configuration for model factory
        # Don't pass source name for benchmark prefixing since benchmarks already include source
        source_config = {
            'name': '',  # Empty to avoid double-prefixing benchmarks
            'source_url': metadata_config.get('source_url', ''),
            'description': metadata_config.get('description', f'{self.name} file scraper')
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
    
    def _save_raw_data(self, raw_rows: List[Dict[str, Any]], source_type: str):
        """Save raw scraped data for debugging.
        
        Args:
            raw_rows: Raw data to save
            source_type: Type of source processed
        """
        try:
            # Create data structure for saving
            save_data = {
                'source_type': source_type,
                'scraper_name': self.name,
                'row_count': len(raw_rows),
                'raw_data': raw_rows
            }
            
            # Save using base class method
            filename = f"file_{self.name}_latest"
            self.save_raw_data(save_data, filename)
            
        except Exception as e:
            self.logger.warning(f"Failed to save raw data: {e}")
    
    def validate_configuration(self) -> List[str]:
        """Validate scraper configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if 'type' not in self.config:
            errors.append("Missing required field: type")
        
        if 'source' not in self.config:
            errors.append("Missing required field: source")
        
        if 'parser' not in self.config:
            errors.append("Missing required field: parser")
        
        # Validate source configuration
        source_config = self.config.get('source', {})
        source_type = self.config.get('type', '')
        
        if source_type == 'csv_file':
            if 'paths' not in source_config and 'path' not in source_config:
                errors.append("CSV file scraper requires 'paths' or 'path' in source config")
        elif source_type in ['html_file', 'markdown_file']:
            if 'file_path' not in source_config and 'path' not in source_config:
                errors.append(f"{source_type} scraper requires 'file_path' or 'path' in source config")
        
        # Validate parser configuration
        parser_config = self.config.get('parser', {})
        if 'columns' not in parser_config and 'column_detection' not in parser_config:
            errors.append("Parser config requires either 'columns' or 'column_detection'")
        
        return errors