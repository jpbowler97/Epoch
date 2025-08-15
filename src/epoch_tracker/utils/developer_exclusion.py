"""
Developer exclusion criteria management for model filtering and FLOP capping.

This module provides functionality to manage exclusion criteria for AI model developers
based on resource constraints (computational budget, GPU access limitations) and
apply FLOP estimate capping while preserving original estimates for transparency.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..models import ConfidenceLevel, EstimationMethod


class DeveloperExclusion:
    """Manages developer exclusion criteria and resource-based FLOP capping logic."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the developer exclusion manager.
        
        Args:
            config_path: Path to exclusion criteria configuration file. If None, uses default location.
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is None:
            # Default path relative to project root
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "developer_exclusion_criteria.json"
        
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load exclusion criteria configuration from JSON file.
        
        Returns:
            Dictionary containing exclusion criteria configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.logger.info(f"Loaded developer exclusion configuration from {self.config_path}")
            return config
            
        except FileNotFoundError:
            self.logger.error(f"Developer exclusion config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in exclusion config file: {e}")
            raise
    
    def is_resource_constrained(self, developer: str) -> bool:
        """Check if a developer is resource-constrained (excluded from high FLOP estimates).
        
        Args:
            developer: Developer/organization name
            
        Returns:
            True if developer is excluded due to resource constraints
        """
        if not developer:
            return True  # Unknown developers are excluded
            
        exclusion_list = self.config.get('exclusion_criteria', {})
        
        # Direct match
        if developer in exclusion_list:
            return exclusion_list[developer].get('excluded', False)
            
        # Case-insensitive search
        developer_lower = developer.lower()
        for name, info in exclusion_list.items():
            if name.lower() == developer_lower:
                return info.get('excluded', False)
                
        # Default policy for unknown developers
        return self.config.get('settings', {}).get('default_exclusion_status', False)
    
    def get_exclusion_reason(self, developer: str) -> Optional[str]:
        """Get the reason why a developer is excluded.
        
        Args:
            developer: Developer/organization name
            
        Returns:
            Exclusion reason string, or None if not excluded
        """
        if not self.is_resource_constrained(developer):
            return None
            
        exclusion_list = self.config.get('exclusion_criteria', {})
        
        # Direct match
        if developer in exclusion_list:
            return exclusion_list[developer].get('reason', 'Resource constraints')
            
        # Case-insensitive search
        developer_lower = developer.lower()
        for name, info in exclusion_list.items():
            if name.lower() == developer_lower:
                return info.get('reason', 'Resource constraints')
                
        return 'Unknown developer - insufficient resource information'
    
    def apply_resource_cap_if_needed(self, developer: str, flop_estimate: float, 
                                   confidence: ConfidenceLevel, method: EstimationMethod, 
                                   reasoning: str) -> Tuple[float, ConfidenceLevel, str, bool]:
        """Apply resource-based FLOP cap if developer is resource-constrained.
        
        Args:
            developer: Developer/organization name
            flop_estimate: Original FLOP estimate
            confidence: Original confidence level
            method: Estimation method used
            reasoning: Original reasoning
            
        Returns:
            Tuple of (final_flop, final_confidence, final_reasoning, was_capped)
        """
        if not self.is_resource_constrained(developer):
            return flop_estimate, confidence, reasoning, False
            
        settings = self.config.get('settings', {})
        cap_value = settings.get('cap_value', 9.9e24)
        
        if flop_estimate <= cap_value:
            return flop_estimate, confidence, reasoning, False
            
        # Apply the cap
        exclusion_reason = self.get_exclusion_reason(developer)
        cap_template = settings.get('cap_reasoning_template', 
                                   'FLOP estimate capped at {cap_value:.1e} due to resource constraint policy: {reason}')
        
        final_reasoning = cap_template.format(cap_value=cap_value, reason=exclusion_reason)
        
        # Optionally reduce confidence when capping
        final_confidence = confidence
        if settings.get('reduce_confidence_on_cap', True):
            if confidence == ConfidenceLevel.HIGH:
                final_confidence = ConfidenceLevel.MEDIUM
            elif confidence == ConfidenceLevel.MEDIUM:
                final_confidence = ConfidenceLevel.LOW
        
        self.logger.info(f"Applied resource cap to {developer}: {flop_estimate:.2e} → {cap_value:.2e} FLOP")
        
        return cap_value, final_confidence, final_reasoning, True
    
    def should_preserve_original(self) -> bool:
        """Check if original estimates should be preserved when capping.
        
        Returns:
            True if original estimates should be stored as alternatives
        """
        return self.config.get('settings', {}).get('preserve_original_estimates', True)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about exclusion criteria.
        
        Returns:
            Dictionary with exclusion statistics
        """
        exclusion_list = self.config.get('exclusion_criteria', {})
        
        total = len(exclusion_list)
        excluded = sum(1 for info in exclusion_list.values() if info.get('excluded', False))
        allowed = total - excluded
        
        return {
            'total_developers': total,
            'excluded_count': excluded,
            'allowed_count': allowed,
            'pending_count': len(self.config.get('pending_review', []))
        }
    
    def add_developer(self, developer: str, excluded: bool, reason: str) -> None:
        """Add or update a developer in the exclusion criteria.
        
        Args:
            developer: Developer/organization name
            excluded: Whether developer should be excluded
            reason: Reason for exclusion/inclusion decision
        """
        exclusion_list = self.config.setdefault('exclusion_criteria', {})
        
        exclusion_list[developer] = {
            'excluded': excluded,
            'reason': reason,
            'last_reviewed': datetime.now().isoformat()
        }
        
        # Update statistics
        self.config['statistics'] = self.get_statistics()
        
        self._save_config()
        self.logger.info(f"{'Excluded' if excluded else 'Allowed'} developer: {developer} - {reason}")
    
    def remove_developer(self, developer: str) -> bool:
        """Remove a developer from exclusion criteria.
        
        Args:
            developer: Developer/organization name
            
        Returns:
            True if developer was found and removed
        """
        exclusion_list = self.config.get('exclusion_criteria', {})
        
        if developer in exclusion_list:
            del exclusion_list[developer]
            self.config['statistics'] = self.get_statistics()
            self._save_config()
            self.logger.info(f"Removed developer from exclusion criteria: {developer}")
            return True
            
        return False
    
    def list_excluded_developers(self) -> List[str]:
        """Get list of all excluded developers.
        
        Returns:
            List of developer names that are excluded
        """
        exclusion_list = self.config.get('exclusion_criteria', {})
        return [name for name, info in exclusion_list.items() if info.get('excluded', False)]
    
    def list_allowed_developers(self) -> List[str]:
        """Get list of all explicitly allowed developers.
        
        Returns:
            List of developer names that are explicitly allowed
        """
        exclusion_list = self.config.get('exclusion_criteria', {})
        return [name for name, info in exclusion_list.items() if not info.get('excluded', False)]
    
    def _save_config(self) -> None:
        """Save current configuration to JSON file."""
        try:
            # Update timestamp
            self.config['last_updated'] = datetime.now().isoformat()
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved exclusion criteria configuration to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save exclusion criteria config: {e}")
            raise


# Legacy alias for backwards compatibility
DeveloperBlacklist = DeveloperExclusion