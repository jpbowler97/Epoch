"""
Developer blacklist management for model filtering and FLOP capping.

This module provides functionality to manage a blacklist of AI model developers
and apply FLOP estimate capping for blacklisted companies while preserving
original estimates for transparency.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..models import ConfidenceLevel, EstimationMethod


class DeveloperBlacklist:
    """Manages developer blacklist configuration and FLOP capping logic."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the developer blacklist manager.
        
        Args:
            config_path: Path to blacklist configuration file. If None, uses default location.
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is None:
            # Default path relative to project root
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "developer_blacklist.json"
        
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load blacklist configuration from JSON file.
        
        Returns:
            Dictionary containing blacklist configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.logger.info(f"Loaded developer blacklist configuration from {self.config_path}")
            self.logger.info(f"Blacklist contains {len(config.get('blacklist', {}))} developers, "
                           f"{config.get('statistics', {}).get('blacklisted_count', 0)} blacklisted")
            
            return config
            
        except FileNotFoundError:
            self.logger.error(f"Developer blacklist config not found at {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in developer blacklist config: {e}")
            raise
    
    def save_config(self) -> None:
        """Save current configuration back to file."""
        try:
            # Update timestamp
            self.config['last_updated'] = datetime.now().isoformat()
            
            # Update statistics
            blacklist = self.config.get('blacklist', {})
            blacklisted_count = sum(1 for dev_info in blacklist.values() if dev_info.get('blacklisted', False))
            
            self.config['statistics'] = {
                'total_developers': len(blacklist),
                'blacklisted_count': blacklisted_count,
                'allowed_count': len(blacklist) - blacklisted_count,
                'pending_count': len(self.config.get('pending_review', []))
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved developer blacklist configuration to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save developer blacklist config: {e}")
            raise
    
    def is_blacklisted(self, developer: str) -> bool:
        """Check if a developer is blacklisted.
        
        Args:
            developer: Developer/company name to check
            
        Returns:
            True if developer is blacklisted, False otherwise
        """
        if not developer:
            return True  # Treat empty/None as blacklisted
        
        blacklist = self.config.get('blacklist', {})
        dev_info = blacklist.get(developer, {})
        
        return dev_info.get('blacklisted', False)
    
    def get_blacklist_reason(self, developer: str) -> str:
        """Get the reason why a developer is blacklisted.
        
        Args:
            developer: Developer/company name
            
        Returns:
            Blacklist reason string, empty if not blacklisted
        """
        if not developer:
            return "Unknown or missing developer information"
        
        blacklist = self.config.get('blacklist', {})
        dev_info = blacklist.get(developer, {})
        
        if dev_info.get('blacklisted', False):
            return dev_info.get('reason', 'No specific reason provided')
        
        return ""
    
# get_blacklist_evidence method removed - consolidated into get_blacklist_reason
    
    def get_cap_value(self) -> float:
        """Get the FLOP cap value for blacklisted developers.
        
        Returns:
            FLOP cap value (default: 9.9e24)
        """
        settings = self.config.get('settings', {})
        return settings.get('cap_value', 9.9e24)
    
    def should_preserve_original(self) -> bool:
        """Check if original estimates should be preserved in alternatives.
        
        Returns:
            True if original estimates should be stored
        """
        settings = self.config.get('settings', {})
        return settings.get('preserve_original_estimates', True)
    
    def should_reduce_confidence_on_cap(self) -> bool:
        """Check if confidence should be reduced when capping is applied.
        
        Returns:
            True if confidence should be reduced for capped estimates
        """
        settings = self.config.get('settings', {})
        return settings.get('reduce_confidence_on_cap', True)
    
    def apply_cap_if_needed(self, developer: str, original_flop: float, 
                           original_confidence: ConfidenceLevel, 
                           original_method: EstimationMethod,
                           original_reasoning: str) -> Tuple[float, ConfidenceLevel, str, bool]:
        """Apply FLOP capping if developer is blacklisted and estimate >= 1e25.
        
        Args:
            developer: Developer/company name
            original_flop: Original FLOP estimate
            original_confidence: Original confidence level
            original_method: Original estimation method
            original_reasoning: Original reasoning text
            
        Returns:
            Tuple of (final_flop, final_confidence, final_reasoning, was_capped)
        """
        # Check if developer is blacklisted and estimate is above threshold
        if not self.is_blacklisted(developer) or original_flop < 1e25:
            return original_flop, original_confidence, original_reasoning, False
        
        # Apply capping
        cap_value = self.get_cap_value()
        blacklist_reason = self.get_blacklist_reason(developer)
        
        # Build capped reasoning
        settings = self.config.get('settings', {})
        template = settings.get('cap_reasoning_template', 
                               "FLOP estimate capped at {cap_value:.1e} due to developer blacklist policy: {reason}")
        
        cap_explanation = template.format(cap_value=cap_value, reason=blacklist_reason)
        
        # Combine with original reasoning for full context
        final_reasoning = f"{cap_explanation}. Original estimate: {original_flop:.2e} FLOP ({original_reasoning})"
        
        # Reduce confidence if configured to do so
        final_confidence = original_confidence
        if self.should_reduce_confidence_on_cap():
            # Reduce confidence by one level, minimum SPECULATIVE
            confidence_levels = [ConfidenceLevel.SPECULATIVE, ConfidenceLevel.LOW, 
                               ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]
            current_index = confidence_levels.index(original_confidence)
            if current_index > 0:
                final_confidence = confidence_levels[current_index - 1]
        
        self.logger.info(f"Applied FLOP cap to {developer} model: {original_flop:.2e} -> {cap_value:.2e}")
        
        return cap_value, final_confidence, final_reasoning, True
    
    def add_pending_developer(self, developer: str) -> None:
        """Add a new developer to the pending review list.
        
        Args:
            developer: Developer/company name to add for review
        """
        pending = self.config.setdefault('pending_review', [])
        
        if developer not in pending and developer not in self.config.get('blacklist', {}):
            pending.append(developer)
            self.logger.info(f"Added {developer} to pending review list")
    
    def get_pending_review(self) -> List[str]:
        """Get list of developers pending review.
        
        Returns:
            List of developer names awaiting blacklist decision
        """
        return self.config.get('pending_review', [])
    
    def get_blacklisted_developers(self) -> List[str]:
        """Get list of currently blacklisted developers.
        
        Returns:
            List of blacklisted developer names
        """
        blacklist = self.config.get('blacklist', {})
        return [dev for dev, info in blacklist.items() if info.get('blacklisted', False)]
    
    def get_allowed_developers(self) -> List[str]:
        """Get list of currently allowed developers.
        
        Returns:
            List of allowed developer names
        """
        blacklist = self.config.get('blacklist', {})
        return [dev for dev, info in blacklist.items() if not info.get('blacklisted', False)]
    
    def update_developer_status(self, developer: str, blacklisted: bool, 
                               reason: str = "") -> None:
        """Update the blacklist status of a developer.
        
        Args:
            developer: Developer/company name
            blacklisted: Whether to blacklist the developer
            reason: Reason for the decision (complete justification)
        """
        blacklist = self.config.setdefault('blacklist', {})
        
        blacklist[developer] = {
            'blacklisted': blacklisted,
            'reason': reason,
            'last_reviewed': datetime.now().isoformat()
        }
        
        # Remove from pending review if present
        pending = self.config.get('pending_review', [])
        if developer in pending:
            pending.remove(developer)
        
        self.logger.info(f"Updated {developer}: blacklisted={blacklisted}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blacklist statistics.
        
        Returns:
            Dictionary with blacklist statistics
        """
        return self.config.get('statistics', {})
    
    def validate_configuration(self) -> List[str]:
        """Validate the blacklist configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required top-level keys
        required_keys = ['blacklist', 'settings']
        for key in required_keys:
            if key not in self.config:
                errors.append(f"Missing required configuration key: {key}")
        
        # Validate blacklist entries
        blacklist = self.config.get('blacklist', {})
        for developer, info in blacklist.items():
            if not isinstance(info, dict):
                errors.append(f"Invalid blacklist entry for {developer}: must be a dictionary")
                continue
            
            # Check required fields
            required_fields = ['blacklisted', 'reason', 'last_reviewed']
            for field in required_fields:
                if field not in info:
                    errors.append(f"Missing field '{field}' for developer {developer}")
        
        # Validate settings
        settings = self.config.get('settings', {})
        if 'cap_value' in settings:
            cap_value = settings['cap_value']
            if not isinstance(cap_value, (int, float)) or cap_value <= 0:
                errors.append("cap_value must be a positive number")
        
        return errors