"""HTTP client utilities for web scraping."""

import time
from typing import Dict, Optional, Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import get_settings, get_config


class HTTPClient:
    """HTTP client with rate limiting and retry logic."""
    
    def __init__(self, base_url: str = "", delay: Optional[float] = None):
        """Initialize HTTP client.
        
        Args:
            base_url: Base URL for relative requests
            delay: Delay between requests (uses config default if None)
        """
        self.base_url = base_url
        self.delay = delay or get_config("scraping.default_delay", 1.0)
        self.last_request_time = 0
        
        settings = get_settings()
        
        # Configure session with retries
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": settings.user_agent,
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        
        # Add Hugging Face token if available and this is a HF request
        if settings.huggingface_token and "huggingface.co" in base_url:
            self.session.headers["Authorization"] = f"Bearer {settings.huggingface_token}"
    
    def _rate_limit(self):
        """Implement rate limiting."""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a GET request with rate limiting.
        
        Args:
            url: URL to request (absolute or relative to base_url)
            **kwargs: Additional arguments for requests.get
            
        Returns:
            Response object
        """
        self._rate_limit()
        
        # Make URL absolute if base_url is set
        if self.base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(self.base_url, url)
        
        # Set default timeout from settings
        if 'timeout' not in kwargs:
            kwargs['timeout'] = get_settings().timeout
        
        response = self.session.get(url, **kwargs)
        response.raise_for_status()
        
        return response
    
    def get_json(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make a GET request and return JSON response.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments for requests.get
            
        Returns:
            JSON response as dictionary
        """
        response = self.get(url, **kwargs)
        return response.json()
    
    def close(self):
        """Close the session."""
        self.session.close()