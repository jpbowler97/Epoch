"""Column mapping utilities for flexible data extraction."""

import logging
from typing import Dict, List, Optional, Union, Any
import re


class ColumnMapper:
    """Utility for mapping column references to data values with flexible matching."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def map_headers_to_indices(self, headers: List[str], 
                              column_patterns: Dict[str, List[str]]) -> Dict[str, int]:
        """Map logical column names to indices based on header patterns.
        
        Args:
            headers: List of column headers from data source
            column_patterns: Dict of logical_name -> [pattern1, pattern2, ...]
            
        Returns:
            Dict mapping logical names to column indices
        """
        column_map = {}
        
        for logical_name, patterns in column_patterns.items():
            index = self._find_header_index(headers, patterns)
            if index is not None:
                column_map[logical_name] = index
                self.logger.debug(f"Mapped '{logical_name}' to column {index} ('{headers[index]}')")
            else:
                self.logger.warning(f"Could not find column for '{logical_name}' with patterns: {patterns}")
        
        return column_map
    
    def _find_header_index(self, headers: List[str], patterns: List[str]) -> Optional[int]:
        """Find the index of a header matching any of the given patterns.
        
        Args:
            headers: List of header strings
            patterns: List of patterns to match (case-insensitive)
            
        Returns:
            Index of first matching header or None
        """
        for i, header in enumerate(headers):
            header_lower = header.lower().strip()
            
            for pattern in patterns:
                pattern_lower = pattern.lower().strip()
                
                # Exact match
                if header_lower == pattern_lower:
                    return i
                
                # Substring match
                if pattern_lower in header_lower:
                    return i
                
                # Word boundary match (for partial matches)
                if re.search(rf'\b{re.escape(pattern_lower)}\b', header_lower):
                    return i
        
        return None
    
    def extract_value(self, row_data: Union[List, Dict], 
                     column_ref: Union[str, int], 
                     default: Any = None) -> Any:
        """Extract value from row data using flexible column reference.
        
        Args:
            row_data: List of values or dict with column names
            column_ref: Column index (int) or column name (str)
            default: Default value if extraction fails
            
        Returns:
            Extracted value or default
        """
        try:
            if isinstance(row_data, dict):
                # Dictionary-based data (e.g., CSV with headers)
                if isinstance(column_ref, str):
                    return row_data.get(column_ref, default)
                elif isinstance(column_ref, int):
                    # Convert index to key if possible
                    keys = list(row_data.keys())
                    if 0 <= column_ref < len(keys):
                        return row_data.get(keys[column_ref], default)
                    return default
            
            elif isinstance(row_data, (list, tuple)):
                # List-based data (e.g., HTML table rows)
                if isinstance(column_ref, int):
                    if 0 <= column_ref < len(row_data):
                        return row_data[column_ref]
                    return default
                elif isinstance(column_ref, str):
                    # Can't use string reference with list data
                    self.logger.warning(f"Cannot use string column reference '{column_ref}' with list data")
                    return default
            
            return default
            
        except Exception as e:
            self.logger.debug(f"Failed to extract value with reference '{column_ref}': {e}")
            return default
    
    def extract_multiple_values(self, row_data: Union[List, Dict], 
                               column_refs: Dict[str, Union[str, int]]) -> Dict[str, Any]:
        """Extract multiple values from row data.
        
        Args:
            row_data: Source data (list or dict)
            column_refs: Dict of field_name -> column_reference
            
        Returns:
            Dict of field_name -> extracted_value
        """
        results = {}
        
        for field_name, column_ref in column_refs.items():
            value = self.extract_value(row_data, column_ref)
            if value is not None:
                results[field_name] = value
        
        return results
    
    def normalize_column_name(self, column_name: str) -> str:
        """Normalize column name for consistent mapping.
        
        Args:
            column_name: Original column name
            
        Returns:
            Normalized column name
        """
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^\w\s]', '', column_name.lower())
        # Replace spaces with underscores
        normalized = re.sub(r'\s+', '_', normalized.strip())
        return normalized
    
    def validate_required_columns(self, column_map: Dict[str, int], 
                                 required_columns: List[str]) -> List[str]:
        """Validate that all required columns are mapped.
        
        Args:
            column_map: Mapping of logical names to indices
            required_columns: List of required logical column names
            
        Returns:
            List of missing required columns
        """
        missing = []
        for required in required_columns:
            if required not in column_map:
                missing.append(required)
        
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
        
        return missing
    
    def auto_detect_columns(self, headers: List[str]) -> Dict[str, int]:
        """Auto-detect common column types from headers.
        
        Args:
            headers: List of column headers
            
        Returns:
            Dict mapping detected column types to indices
        """
        common_patterns = {
            'model': ['model', 'name', 'model_name', 'model name'],
            'developer': ['developer', 'organization', 'org', 'company', 'author'],
            'score': ['score', 'overall', 'total', 'rating', 'result'],
            'rank': ['rank', 'position', 'place', '#'],
            'elo': ['elo', 'rating', 'arena elo', 'arena_elo'],
            'coding': ['coding', 'code', 'programming'],
            'math': ['math', 'mathematics', 'mathematical'],
            'reasoning': ['reasoning', 'logic', 'logical'],
            'license': ['license', 'licence', 'licensing'],
            'votes': ['votes', 'vote', 'count', 'num_votes'],
            'parameters': ['parameters', 'params', 'param_count', 'size'],
            'date': ['date', 'release_date', 'created', 'updated']
        }
        
        return self.map_headers_to_indices(headers, common_patterns)