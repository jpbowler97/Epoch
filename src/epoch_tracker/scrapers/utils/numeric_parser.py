"""Shared numeric parsing utilities for scrapers."""

import re
import logging
from typing import Optional, Union
import pandas as pd


class NumericParser:
    """Utility class for consistent numeric parsing across all scrapers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_numeric(self, value: Union[str, int, float, None]) -> Optional[float]:
        """Parse a numeric value from various input types.
        
        Handles:
        - String formatting (commas, percentages, plus signs)
        - Different numeric types
        - NaN and empty values
        - Error cases with logging
        
        Args:
            value: Input value to parse
            
        Returns:
            Parsed float value or None if parsing fails
        """
        if value is None or pd.isna(value):
            return None
        
        # Handle already numeric types
        if isinstance(value, (int, float)):
            return float(value) if not pd.isna(value) else None
        
        # Convert to string and clean
        str_value = str(value).strip()
        if not str_value or str_value.lower() in ['', 'nan', 'null', 'none', '-', 'n/a']:
            return None
        
        try:
            # Remove common formatting
            cleaned = self._clean_numeric_string(str_value)
            if not cleaned:
                return None
                
            return float(cleaned)
            
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Failed to parse numeric value '{value}': {e}")
            return None
    
    def _clean_numeric_string(self, value: str) -> str:
        """Clean a string for numeric parsing.
        
        Args:
            value: String to clean
            
        Returns:
            Cleaned string ready for float conversion
        """
        # Remove common numeric formatting
        patterns_to_remove = [
            r'[,\s]',          # Commas and whitespace
            r'[+](?=\d)',      # Plus signs before numbers
            r'%(?=\s*$)',      # Trailing percentage signs
            r'[^\d.-]'         # Anything that's not digit, dot, or minus
        ]
        
        cleaned = value
        for pattern in patterns_to_remove[:-1]:  # Apply first 3 patterns
            cleaned = re.sub(pattern, '', cleaned)
        
        # Special handling for the last pattern to preserve decimals and negatives
        if re.match(r'^-?\d+\.?\d*$', cleaned):
            return cleaned
        
        # If still not clean, try more aggressive cleaning
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)
        if numbers:
            return numbers[0]
        
        return ''
    
    def parse_percentage(self, value: Union[str, int, float, None]) -> Optional[float]:
        """Parse a percentage value and convert to decimal.
        
        Args:
            value: Input percentage value
            
        Returns:
            Decimal value (e.g., 95.5% -> 0.955) or None
        """
        numeric_value = self.parse_numeric(value)
        if numeric_value is None:
            return None
            
        # If the original value contained %, assume it's already in percentage form
        if isinstance(value, str) and '%' in value:
            return numeric_value / 100.0
        
        # If value is > 1, assume it's percentage form
        if numeric_value > 1:
            return numeric_value / 100.0
            
        return numeric_value
    
    def parse_score_range(self, value: Union[str, None], 
                         expected_min: float = 0.0, 
                         expected_max: float = 100.0) -> Optional[float]:
        """Parse a score with validation against expected range.
        
        Args:
            value: Score value to parse
            expected_min: Minimum expected value
            expected_max: Maximum expected value
            
        Returns:
            Parsed score or None if outside expected range
        """
        score = self.parse_numeric(value)
        if score is None:
            return None
        
        if expected_min <= score <= expected_max:
            return score
        
        self.logger.warning(f"Score {score} outside expected range [{expected_min}, {expected_max}]")
        return score  # Return anyway but log warning
    
    def parse_rank(self, value: Union[str, int, None]) -> Optional[int]:
        """Parse ranking/position values.
        
        Args:
            value: Rank value to parse
            
        Returns:
            Integer rank or None
        """
        if value is None:
            return None
        
        # Handle string ranks like "1st", "2nd", "#5"
        if isinstance(value, str):
            # Extract first number from string
            numbers = re.findall(r'\d+', value.strip())
            if numbers:
                return int(numbers[0])
            return None
        
        # Handle numeric ranks
        try:
            rank = int(value)
            return rank if rank > 0 else None
        except (ValueError, TypeError):
            return None