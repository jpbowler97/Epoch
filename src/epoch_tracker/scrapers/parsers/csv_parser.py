"""CSV file parsing with flexible column mapping and file discovery."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import glob

from ..utils.column_mapper import ColumnMapper
from ..utils.numeric_parser import NumericParser


class CSVParser:
    """Parser for CSV files with intelligent file discovery and column mapping."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CSV parser with configuration.
        
        Args:
            config: Parser configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.column_mapper = ColumnMapper()
        self.numeric_parser = NumericParser()
        
        # CSV settings
        self.encoding = config.get('encoding', 'utf-8')
        self.delimiter = config.get('delimiter', ',')
        self.skip_rows = config.get('skip_rows', 0)
        
    def parse_files(self, file_config: Dict[str, Any], 
                   parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse CSV files based on configuration.
        
        Args:
            file_config: File location and discovery configuration
            parser_config: Parsing and column mapping configuration
            
        Returns:
            List of parsed row data
        """
        # Discover files
        csv_files = self._discover_files(file_config)
        
        if not csv_files:
            self.logger.warning("No CSV files found")
            return []
        
        # Parse files in priority order
        all_rows = []
        for csv_file in csv_files:
            try:
                rows = self._parse_single_file(csv_file, parser_config)
                all_rows.extend(rows)
                
                # If we found data and only want first file, break
                if rows and file_config.get('use_first_only', True):
                    self.logger.info(f"Using data from first available file: {csv_file}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Failed to parse CSV file {csv_file}: {e}")
                continue
        
        return all_rows
    
    def _discover_files(self, file_config: Dict[str, Any]) -> List[Path]:
        """Discover CSV files based on configuration.
        
        Args:
            file_config: File discovery configuration
            
        Returns:
            List of CSV files in priority order
        """
        files = []
        
        # Primary files (highest priority)
        primary_paths = file_config.get('paths', [])
        for path_str in primary_paths:
            path = Path(path_str)
            if not path.is_absolute():
                # Make relative to project root
                project_root = Path(__file__).parent.parent.parent.parent.parent
                path = project_root / path
            
            if path.exists() and path.is_file():
                files.append(path)
                self.logger.debug(f"Found primary file: {path}")
        
        # Fallback pattern matching
        fallback_pattern = file_config.get('fallback_pattern')
        if fallback_pattern and not files:
            pattern_path = Path(fallback_pattern)
            if not pattern_path.is_absolute():
                project_root = Path(__file__).parent.parent.parent.parent.parent
                pattern_path = project_root / pattern_path
            
            # Use glob to find matching files
            matching_files = list(pattern_path.parent.glob(pattern_path.name))
            
            # Sort by modification time (newest first)
            matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            files.extend(matching_files)
            
            if matching_files:
                self.logger.info(f"Found {len(matching_files)} files matching fallback pattern")
        
        return files
    
    def _parse_single_file(self, file_path: Path, 
                          parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a single CSV file.
        
        Args:
            file_path: Path to CSV file
            parser_config: Parsing configuration
            
        Returns:
            List of parsed row data
        """
        self.logger.info(f"Parsing CSV file: {file_path}")
        
        try:
            # Try pandas first for better encoding detection
            df = pd.read_csv(
                file_path, 
                encoding=self.encoding,
                delimiter=self.delimiter,
                skiprows=self.skip_rows
            )
            
            # Convert to list of dictionaries
            return self._parse_dataframe(df, parser_config)
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    self.logger.info(f"Trying encoding: {encoding}")
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        delimiter=self.delimiter,
                        skiprows=self.skip_rows
                    )
                    return self._parse_dataframe(df, parser_config)
                except Exception:
                    continue
            
            # Fall back to basic CSV reader
            return self._parse_with_csv_reader(file_path, parser_config)
            
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path} with pandas: {e}")
            return self._parse_with_csv_reader(file_path, parser_config)
    
    def _parse_dataframe(self, df: pd.DataFrame, 
                        parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse pandas DataFrame into row data.
        
        Args:
            df: Pandas DataFrame
            parser_config: Parsing configuration
            
        Returns:
            List of parsed row data
        """
        # Map columns
        columns_config = parser_config.get('columns', {})
        column_detection = parser_config.get('column_detection', {})
        
        # If no explicit column detection, try to map from columns config
        if not column_detection and columns_config:
            column_detection = {k: [v] for k, v in columns_config.items()}
        
        column_map = self.column_mapper.map_headers_to_indices(df.columns.tolist(), column_detection)
        
        if not column_map:
            self.logger.warning("No columns could be mapped")
            return []
        
        # Process rows
        parsed_rows = []
        for _, row in df.iterrows():
            try:
                row_data = self._parse_dataframe_row(row, column_map, parser_config)
                if row_data:
                    parsed_rows.append(row_data)
            except Exception as e:
                self.logger.debug(f"Error parsing dataframe row: {e}")
                continue
        
        self.logger.info(f"Parsed {len(parsed_rows)} rows from DataFrame")
        return parsed_rows
    
    def _parse_dataframe_row(self, row: pd.Series, column_map: Dict[str, int], 
                           parser_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single DataFrame row.
        
        Args:
            row: Pandas Series (DataFrame row)
            column_map: Column mapping
            parser_config: Parser configuration
            
        Returns:
            Parsed row data or None
        """
        row_data = {}
        
        # Extract mapped values
        for logical_name, column_index in column_map.items():
            if column_index < len(row):
                raw_value = row.iloc[column_index]
                
                # Handle pandas NaN
                if pd.isna(raw_value):
                    continue
                
                # Parse based on field type
                if logical_name in ['score', 'elo', 'coding', 'math', 'reasoning', 'votes', 'rank']:
                    parsed_value = self.numeric_parser.parse_numeric(raw_value)
                else:
                    parsed_value = str(raw_value).strip() if raw_value else None
                
                if parsed_value is not None and parsed_value != '':
                    row_data[logical_name] = parsed_value
        
        # Apply filters and validation
        return self._validate_and_filter_row(row_data, parser_config)
    
    def _parse_with_csv_reader(self, file_path: Path, 
                              parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse CSV using basic csv.DictReader as fallback.
        
        Args:
            file_path: Path to CSV file
            parser_config: Parser configuration
            
        Returns:
            List of parsed row data
        """
        self.logger.info(f"Using csv.DictReader for {file_path}")
        
        parsed_rows = []
        
        try:
            with open(file_path, 'r', encoding=self.encoding, newline='') as f:
                # Skip rows if needed
                for _ in range(self.skip_rows):
                    f.readline()
                
                reader = csv.DictReader(f, delimiter=self.delimiter)
                
                for row_dict in reader:
                    try:
                        row_data = self._parse_dict_row(row_dict, parser_config)
                        if row_data:
                            parsed_rows.append(row_data)
                    except Exception as e:
                        self.logger.debug(f"Error parsing CSV row: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path} with csv.DictReader: {e}")
            return []
        
        self.logger.info(f"Parsed {len(parsed_rows)} rows with csv.DictReader")
        return parsed_rows
    
    def _parse_dict_row(self, row_dict: Dict[str, str], 
                       parser_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a row from csv.DictReader.
        
        Args:
            row_dict: Dictionary from csv.DictReader
            parser_config: Parser configuration
            
        Returns:
            Parsed row data or None
        """
        columns_config = parser_config.get('columns', {})
        row_data = {}
        
        # Extract values using column mapping
        for logical_name, column_name in columns_config.items():
            if column_name in row_dict:
                raw_value = row_dict[column_name]
                
                # Parse based on field type
                if logical_name in ['score', 'elo', 'coding', 'math', 'reasoning', 'votes', 'rank']:
                    parsed_value = self.numeric_parser.parse_numeric(raw_value)
                else:
                    parsed_value = raw_value.strip() if raw_value else None
                
                if parsed_value is not None and parsed_value != '':
                    row_data[logical_name] = parsed_value
        
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
        
        # Score range filters
        min_score = filters.get('min_score')
        if min_score is not None:
            score = row_data.get('score')
            if score is None or score < min_score:
                return None
        
        return row_data