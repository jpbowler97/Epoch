"""Local file parsing for HTML and other static file formats."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from bs4 import BeautifulSoup

from ..utils.column_mapper import ColumnMapper
from ..utils.numeric_parser import NumericParser


class FileParser:
    """Parser for local HTML and static files."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize file parser with configuration.
        
        Args:
            config: Parser configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.column_mapper = ColumnMapper()
        self.numeric_parser = NumericParser()
        
        # File settings
        self.encoding = config.get('encoding', 'utf-8')
    
    def parse_file(self, file_path: Union[str, Path], 
                  parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a local file.
        
        Args:
            file_path: Path to the file
            parser_config: Parsing configuration
            
        Returns:
            List of parsed row data
        """
        file_path = Path(file_path)
        
        if not file_path.is_absolute():
            # Make relative to project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            file_path = project_root / file_path
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return []
        
        # Determine file type and parse
        if file_path.suffix.lower() in ['.html', '.htm']:
            return self._parse_html_file(file_path, parser_config)
        else:
            self.logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
    
    def _parse_html_file(self, file_path: Path, 
                        parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse HTML file to extract table data.
        
        Args:
            file_path: Path to HTML file
            parser_config: Parsing configuration
            
        Returns:
            List of parsed row data
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                self.logger.error(f"Could not decode file {file_path}")
                return []
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return []
        
        # Parse HTML content
        try:
            soup = BeautifulSoup(content, 'html.parser')
        except Exception as e:
            self.logger.error(f"Failed to parse HTML content: {e}")
            return []
        
        # Find and parse table
        table_selector = parser_config.get('table_selector', 'table')
        table = soup.select_one(table_selector)
        
        if not table:
            self.logger.warning(f"No table found with selector '{table_selector}'")
            return []
        
        return self._extract_table_data(table, parser_config)
    
    def _extract_table_data(self, table, parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from HTML table element.
        
        Args:
            table: BeautifulSoup table element
            parser_config: Parsing configuration
            
        Returns:
            List of row data dictionaries
        """
        # Extract headers
        header_row = table.find('tr')
        if not header_row:
            self.logger.error("No header row found in table")
            return []
        
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        self.logger.debug(f"Found table headers: {headers}")
        
        # Map columns
        column_detection = parser_config.get('column_detection', {})
        
        # If no column_detection, try to use columns mapping
        if not column_detection:
            columns_mapping = parser_config.get('columns', {})
            if columns_mapping:
                # Convert columns mapping to column_detection format
                column_detection = {k: [v] for k, v in columns_mapping.items()}
            else:
                # Try to auto-detect common columns
                column_detection = self.column_mapper.auto_detect_columns(headers)
        
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
        
        self.logger.info(f"Extracted {len(parsed_rows)} rows from HTML table")
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
        
        # Remove index column if present (first column is numeric)
        if cell_texts and cell_texts[0].isdigit():
            cell_texts = cell_texts[1:]
            # Adjust column indices
            column_map = {k: v-1 if v > 0 else v for k, v in column_map.items()}
        
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
                
                if parsed_value is not None and parsed_value != '':
                    row_data[logical_name] = parsed_value
        
        # Validate and filter
        return self._validate_and_filter_row(row_data, parser_config)
    
    def _validate_and_filter_row(self, row_data: Dict[str, Any], 
                                parser_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and filter a parsed row.
        
        Args:
            row_data: Parsed row data
            parser_config: Parser configuration
            
        Returns:
            Validated row data or None if filtered out
        """
        # Check required fields
        required_fields = parser_config.get('required_fields', ['model'])
        for field in required_fields:
            if field not in row_data or not row_data[field]:
                return None
        
        # Apply filters
        filters = parser_config.get('filters', {})
        
        # Model name filters
        model_filters = filters.get('model_patterns', [])
        if model_filters:
            model_name = str(row_data.get('model', '')).lower()
            if not any(pattern.lower() in model_name for pattern in model_filters):
                return None
        
        return row_data