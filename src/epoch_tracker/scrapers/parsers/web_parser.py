"""Web-based data parsing for HTTP sources and HTML tables."""

import logging
from typing import Dict, List, Optional, Any
import requests
from bs4 import BeautifulSoup
import time

from ..utils.column_mapper import ColumnMapper
from ..utils.numeric_parser import NumericParser


class WebParser:
    """Parser for web-based data sources with HTML table extraction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize web parser with configuration.
        
        Args:
            config: Parser configuration including HTTP settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.column_mapper = ColumnMapper()
        self.numeric_parser = NumericParser()
        
        # Initialize HTTP session
        self.session = requests.Session()
        
        # Configure headers
        headers = config.get('headers', {})
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        default_headers.update(headers)
        self.session.headers.update(default_headers)
        
        # Request settings
        self.timeout = config.get('timeout', 30)
        self.retry_count = config.get('retry_count', 3)
        self.retry_delay = config.get('retry_delay', 1)
        
    def fetch_and_parse(self, url: str, parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch web content and parse table data.
        
        Args:
            url: URL to fetch
            parser_config: Configuration for parsing the content
            
        Returns:
            List of dictionaries containing parsed row data
        """
        # Fetch content
        content = self._fetch_content(url)
        if not content:
            return []
        
        # Parse content
        return self._parse_html_content(content, parser_config)
    
    def _fetch_content(self, url: str) -> Optional[str]:
        """Fetch content from URL with retry logic.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        for attempt in range(self.retry_count):
            try:
                self.logger.info(f"Fetching content from {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                self.logger.info(f"Successfully fetched {len(response.content)} bytes from {url}")
                return response.text
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                
                if attempt < self.retry_count - 1:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to fetch {url} after {self.retry_count} attempts")
        
        return None
    
    def _parse_html_content(self, content: str, parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse HTML content to extract table data.
        
        Args:
            content: HTML content
            parser_config: Configuration for parsing
            
        Returns:
            List of parsed row data
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
        except Exception as e:
            self.logger.error(f"Failed to parse HTML content: {e}")
            return []
        
        # Find table
        table_selector = parser_config.get('table_selector', 'table')
        table = soup.select_one(table_selector)
        
        if not table:
            self.logger.warning(f"No table found with selector '{table_selector}'")
            return []
        
        return self._extract_table_data(table, parser_config)
    
    def _extract_table_data(self, table, parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from HTML table.
        
        Args:
            table: BeautifulSoup table element
            parser_config: Configuration for extraction
            
        Returns:
            List of row data dictionaries
        """
        # Get headers
        header_row = table.find('tr')
        if not header_row:
            self.logger.error("No header row found in table")
            return []
        
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        self.logger.debug(f"Found table headers: {headers}")
        
        # Map headers to column indices
        column_detection = parser_config.get('column_detection', {})
        column_map = self.column_mapper.map_headers_to_indices(headers, column_detection)
        
        if not column_map:
            self.logger.warning("No columns could be mapped from headers")
            return []
        
        # Extract data rows
        rows = table.find_all('tr')[1:]  # Skip header row
        parsed_rows = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:  # Need minimum columns
                continue
            
            try:
                row_data = self._parse_table_row(cells, column_map, parser_config)
                if row_data:
                    parsed_rows.append(row_data)
            except Exception as e:
                self.logger.debug(f"Error parsing table row: {e}")
                continue
        
        self.logger.info(f"Extracted {len(parsed_rows)} rows from table")
        return parsed_rows
    
    def _parse_table_row(self, cells, column_map: Dict[str, int], 
                         parser_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single table row.
        
        Args:
            cells: List of table cell elements
            column_map: Mapping of logical names to column indices
            parser_config: Parser configuration
            
        Returns:
            Dictionary of parsed row data or None
        """
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        # Extract values using column map
        row_data = {}
        
        for logical_name, column_index in column_map.items():
            if column_index < len(cell_texts):
                raw_value = cell_texts[column_index]
                
                # Parse numeric values for known numeric fields
                if logical_name in ['score', 'elo', 'coding', 'math', 'reasoning', 'votes', 'rank']:
                    parsed_value = self.numeric_parser.parse_numeric(raw_value)
                else:
                    parsed_value = raw_value if raw_value else None
                
                if parsed_value is not None:
                    row_data[logical_name] = parsed_value
        
        # Check for required fields
        required_fields = parser_config.get('required_fields', ['model'])
        for field in required_fields:
            if field not in row_data or not row_data[field]:
                return None
        
        # Apply filters
        filters = parser_config.get('filters', {})
        if not self._apply_filters(row_data, filters):
            return None
        
        return row_data
    
    def _apply_filters(self, row_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to determine if row should be included.
        
        Args:
            row_data: Parsed row data
            filters: Filter configuration
            
        Returns:
            True if row passes filters, False otherwise
        """
        # Model name filters
        model_filters = filters.get('model_patterns', [])
        if model_filters:
            model_name = str(row_data.get('model', '')).lower()
            if not any(pattern.lower() in model_name for pattern in model_filters):
                return False
        
        # Score range filters
        min_score = filters.get('min_score')
        if min_score is not None:
            score = row_data.get('score')
            if score is None or score < min_score:
                return False
        
        max_score = filters.get('max_score')
        if max_score is not None:
            score = row_data.get('score')
            if score is None or score > max_score:
                return False
        
        return True
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()