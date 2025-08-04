"""Markdown table parsing for GitHub-style tables."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd

from ..utils.column_mapper import ColumnMapper
from ..utils.numeric_parser import NumericParser


class MarkdownParser:
    """Parser for GitHub-style markdown tables."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize markdown parser with configuration.
        
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
        """Parse markdown file to extract table data.
        
        Args:
            file_path: Path to markdown file
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
        
        # Read file content
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                self.logger.error(f"Could not decode file {file_path}")
                return []
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return []
        
        # Extract tables
        tables = self._extract_tables(lines)
        
        if not tables:
            self.logger.warning(f"No tables found in {file_path}")
            return []
        
        # Parse all tables and combine results
        all_rows = []
        for i, table_df in enumerate(tables):
            try:
                rows = self._parse_table(table_df, parser_config)
                all_rows.extend(rows)
                self.logger.debug(f"Extracted {len(rows)} rows from table {i+1}")
            except Exception as e:
                self.logger.warning(f"Error parsing table {i+1}: {e}")
                continue
        
        return all_rows
    
    def _extract_tables(self, lines: List[str]) -> List[pd.DataFrame]:
        """Extract all tables from markdown lines.
        
        Args:
            lines: List of lines from markdown file
            
        Returns:
            List of DataFrames representing tables
        """
        tables = []
        current_table = []
        in_table = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('|') and line.endswith('|'):
                # This is a table row
                in_table = True
                # Split and clean cells (remove first and last empty cells)
                cells = [cell.strip() for cell in line.split('|')][1:-1]
                current_table.append(cells)
            else:
                # Not a table row
                if in_table and current_table:
                    # End of current table, process it
                    table_df = self._create_dataframe_from_table(current_table)
                    if table_df is not None:
                        tables.append(table_df)
                    current_table = []
                in_table = False
        
        # Handle table at end of file
        if in_table and current_table:
            table_df = self._create_dataframe_from_table(current_table)
            if table_df is not None:
                tables.append(table_df)
        
        self.logger.info(f"Found {len(tables)} tables in markdown")
        return tables
    
    def _create_dataframe_from_table(self, table_rows: List[List[str]]) -> Optional[pd.DataFrame]:
        """Create DataFrame from table rows.
        
        Args:
            table_rows: List of table rows (each row is list of cells)
            
        Returns:
            DataFrame or None if invalid table
        """
        if len(table_rows) < 2:  # Need at least header + one data row
            return None
        
        # First row is headers
        headers = table_rows[0]
        
        # Skip separator row (usually second row with dashes)
        data_rows = []
        for row in table_rows[1:]:
            # Skip rows that look like separators (all dashes/colons)
            if all(cell.replace('-', '').replace(':', '').strip() == '' for cell in row):
                continue
            data_rows.append(row)
        
        if not data_rows:
            return None
        
        try:
            # Ensure all rows have same number of columns
            max_cols = max(len(headers), max(len(row) for row in data_rows))
            
            # Pad rows to same length
            headers = headers + [''] * (max_cols - len(headers))
            data_rows = [row + [''] * (max_cols - len(row)) for row in data_rows]
            
            return pd.DataFrame(data_rows, columns=headers)
            
        except Exception as e:
            self.logger.debug(f"Failed to create DataFrame from table: {e}")
            return None
    
    def _parse_table(self, table_df: pd.DataFrame, 
                    parser_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a single table DataFrame.
        
        Args:
            table_df: DataFrame representing the table
            parser_config: Parsing configuration
            
        Returns:
            List of parsed row data
        """
        if table_df.empty:
            return []
        
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
                column_detection = self.column_mapper.auto_detect_columns(table_df.columns.tolist())
        
        column_map = self.column_mapper.map_headers_to_indices(table_df.columns.tolist(), column_detection)
        
        if not column_map:
            self.logger.warning("No columns could be mapped from table headers")
            return []
        
        # Apply filters to table if specified
        filters = parser_config.get('filters', {})
        model_filters = filters.get('model_patterns', [])
        
        if model_filters:
            # Filter DataFrame based on model patterns
            mask = pd.Series([False] * len(table_df))
            for col in table_df.columns:
                if table_df[col].dtype == object:  # String columns only
                    for pattern in model_filters:
                        mask = mask | table_df[col].str.contains(pattern, case=False, na=False)
            
            if mask.any():
                table_df = table_df[mask]
            else:
                self.logger.debug("No rows matched filter patterns")
                return []
        
        # Parse rows
        parsed_rows = []
        for _, row in table_df.iterrows():
            try:
                row_data = self._parse_table_row(row, column_map, parser_config)
                if row_data:
                    parsed_rows.append(row_data)
            except Exception as e:
                self.logger.debug(f"Error parsing table row: {e}")
                continue
        
        return parsed_rows
    
    def _parse_table_row(self, row: pd.Series, column_map: Dict[str, int], 
                        parser_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single table row.
        
        Args:
            row: Pandas Series (table row)
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
        
        # Validate required fields
        required_fields = parser_config.get('required_fields', ['model'])
        for field in required_fields:
            if field not in row_data or not row_data[field]:
                return None
        
        return row_data