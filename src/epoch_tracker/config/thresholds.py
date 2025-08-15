"""
Centralized threshold configuration management.

This module provides a single source of truth for all FLOP threshold values
used throughout the epoch_tracker system, preventing inconsistencies from
hardcoded values scattered across the codebase.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


class ThresholdClassification:
    """Standard threshold classification constants used throughout the system."""
    HIGH_CONFIDENCE_ABOVE = 'high_confidence_above_1e25'
    LIKELY_ABOVE = 'likely_above_1e25' 
    UNCERTAIN = 'uncertain'
    LIKELY_BELOW = 'likely_below_1e25'
    HIGH_CONFIDENCE_BELOW = 'high_confidence_below_1e25'

# Classifications that indicate model should be considered as candidate (above/uncertain)
CANDIDATE_CLASSIFICATIONS = [
    ThresholdClassification.HIGH_CONFIDENCE_ABOVE,
    ThresholdClassification.LIKELY_ABOVE,
    ThresholdClassification.UNCERTAIN
]


@dataclass(frozen=True)
class ThresholdConfig:
    """Immutable configuration for FLOP threshold classification.
    
    This class loads threshold values from the central configuration file
    and provides them as a centralized source of truth throughout the system.
    """
    
    high_confidence_above_threshold: float
    high_confidence_below_threshold: float
    target_threshold: float
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'ThresholdConfig':
        """Load threshold configuration from JSON file.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
            
        Returns:
            ThresholdConfig instance with loaded values
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            KeyError: If required threshold values are missing
            ValueError: If threshold values are invalid
        """
        if config_path is None:
            # Default path relative to project root
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "flop_estimation_methods.json"
        
        logger = logging.getLogger(__name__)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            threshold_config = config.get("threshold_classification", {})
            
            # Extract threshold values with validation
            high_above = threshold_config.get("high_confidence_above_threshold")
            target = threshold_config.get("target_threshold")
            
            if high_above is None:
                raise KeyError("Missing 'high_confidence_above_threshold' in config")
            if target is None:
                raise KeyError("Missing 'target_threshold' in config")
            
            # Use target_threshold as the high_confidence_below_threshold
            high_below = target
            
            # Validate threshold relationships
            if high_above <= target:
                raise ValueError(f"high_confidence_above_threshold ({high_above}) must be > target_threshold ({target})")
            # Note: high_below == target is now intentional and valid
            
            logger.info(f"Loaded threshold configuration from {config_path}")
            logger.info(f"  High confidence above: {high_above:.1e} FLOP")
            logger.info(f"  High confidence below: {high_below:.1e} FLOP")
            logger.info(f"  Target threshold: {target:.1e} FLOP")
            
            return cls(
                high_confidence_above_threshold=float(high_above),
                high_confidence_below_threshold=float(high_below),
                target_threshold=float(target)
            )
            
        except FileNotFoundError:
            logger.error(f"Threshold configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in threshold configuration: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid threshold configuration: {e}")
            raise
    
    def get_classification_ranges(self) -> Dict[str, str]:
        """Get human-readable descriptions of classification ranges.
        
        Returns:
            Dictionary mapping classification names to range descriptions
        """
        return {
            'HIGH_CONFIDENCE_ABOVE': f"≥ {self.high_confidence_above_threshold:.1e} FLOP (well above {self.target_threshold:.1e} threshold)",
            'HIGH_CONFIDENCE_BELOW': f"≤ {self.high_confidence_below_threshold:.1e} FLOP (at or below {self.target_threshold:.1e} threshold)",
            'NOT_SURE': f"Between {self.high_confidence_below_threshold:.1e} and {self.high_confidence_above_threshold:.1e} FLOP (uncertain around {self.target_threshold:.1e} threshold)"
        }
    
    def validate_thresholds(self) -> bool:
        """Validate that threshold relationships are correct.
        
        Returns:
            True if all relationships are valid
        """
        return (
            self.high_confidence_below_threshold < self.target_threshold < self.high_confidence_above_threshold
            and self.high_confidence_below_threshold < self.high_confidence_above_threshold
        )


# Global instance - loaded lazily on first access
_threshold_config: Optional[ThresholdConfig] = None


def get_threshold_config(config_path: Optional[Path] = None) -> ThresholdConfig:
    """Get the global threshold configuration instance.
    
    This function provides a convenient way to access threshold configuration
    throughout the codebase. The configuration is loaded once and cached.
    
    Args:
        config_path: Path to configuration file. Only used on first call.
        
    Returns:
        ThresholdConfig instance
    """
    global _threshold_config
    
    if _threshold_config is None:
        _threshold_config = ThresholdConfig.load(config_path)
    
    return _threshold_config


def reset_threshold_config():
    """Reset the cached threshold configuration.
    
    This is primarily useful for testing or when configuration changes at runtime.
    """
    global _threshold_config
    _threshold_config = None