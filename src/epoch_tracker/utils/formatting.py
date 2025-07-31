"""Formatting utilities for Epoch Tracker."""

from typing import Optional, Union
from ..config import get_settings


def format_flop_value(flop: Optional[Union[float, int]], format_type: Optional[str] = None) -> Optional[Union[float, str]]:
    """Format FLOP value according to configuration.
    
    Args:
        flop: FLOP value to format
        format_type: Override format type ("scientific" or "numeric")
        
    Returns:
        Formatted FLOP value or None if input is None
    """
    if flop is None:
        return None
    
    if format_type is None:
        format_type = get_settings().flop_format
    
    if format_type == "scientific":
        # Return as string in scientific notation
        return f"{flop:.2e}"
    else:  # numeric
        # Return as raw number (float/int) - JSON will serialize this as a number
        return float(flop)


def parse_flop_value(flop_input: Union[str, float, int, None]) -> Optional[float]:
    """Parse FLOP value from various input formats.
    
    Args:
        flop_input: FLOP value in string (scientific notation) or numeric format
        
    Returns:
        FLOP value as float or None if input is None/invalid
    """
    if flop_input is None:
        return None
    
    if isinstance(flop_input, (int, float)):
        return float(flop_input)
    
    if isinstance(flop_input, str):
        try:
            return float(flop_input)
        except ValueError:
            return None
    
    return None