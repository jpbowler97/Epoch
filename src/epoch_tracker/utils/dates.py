"""Date parsing and formatting utilities."""

from datetime import datetime
from typing import Optional, Union

from dateutil.parser import parse as dateutil_parse


def parse_date(date_str: Union[str, datetime, None]) -> Optional[datetime]:
    """Parse a date string into a datetime object.
    
    Args:
        date_str: Date string, datetime object, or None
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    if date_str is None:
        return None
    
    if isinstance(date_str, datetime):
        return date_str
    
    if not isinstance(date_str, str):
        return None
    
    try:
        # Try to parse with dateutil (handles many formats)
        return dateutil_parse(date_str)
    except (ValueError, TypeError):
        # Try common formats manually
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%B %Y",  # "March 2024"
            "%B %d, %Y",  # "March 15, 2024"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None


def format_date(dt: Optional[datetime], format_str: str = "%Y-%m-%d") -> Optional[str]:
    """Format a datetime object as a string.
    
    Args:
        dt: Datetime object or None
        format_str: Format string for strftime
        
    Returns:
        Formatted date string or None
    """
    if dt is None:
        return None
    
    return dt.strftime(format_str)


def is_recent(dt: Optional[datetime], days: int = 365) -> bool:
    """Check if a date is within the last N days.
    
    Args:
        dt: Datetime to check
        days: Number of days to consider recent
        
    Returns:
        True if the date is recent, False otherwise
    """
    if dt is None:
        return False
    
    now = datetime.utcnow()
    delta = now - dt
    
    return delta.days <= days