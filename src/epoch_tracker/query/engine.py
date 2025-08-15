"""Core query engine for model data."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from ..models import Model, ModelCollection, ConfidenceLevel
from ..storage import JSONStorage


class ModelQueryEngine:
    """Engine for querying and filtering model data."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the query engine.
        
        Args:
            data_dir: Directory containing data files
        """
        self.logger = logging.getLogger(__name__)
        
        if data_dir is None:
            data_dir = Path("data")
        
        self.data_dir = Path(data_dir)
        self.storage = JSONStorage(data_dir)
        
        self._models_cache: Optional[List[Model]] = None
        self._cache_timestamp: Optional[datetime] = None
    
    def load_data(self, files: Optional[List[str]] = None) -> List[Model]:
        """Load model data from JSON files.
        
        Args:
            files: Specific filenames to load (without .json extension).
                  If None, loads from consolidated estimated_models.json from estimated/ or falls back to scraped files.
                  
        Returns:
            List of loaded models
        """
        # Check cache first
        if self._models_cache is not None and files is None:
            self.logger.debug("Using cached model data")
            return self._models_cache
        
        models = []
        
        if files is None:
            # First try to load from consolidated estimated models file
            try:
                collection = self.storage.load_models("estimated_models", stage="estimated")
                if collection:
                    models.extend(collection.models)
                    self.logger.info(f"Loaded {len(collection.models)} models from consolidated estimated_models.json")
                    
                    # Cache results
                    self._models_cache = models
                    self._cache_timestamp = datetime.utcnow()
                    
                    return models
            except Exception as e:
                self.logger.warning(f"Failed to load consolidated file, falling back to scraped files: {e}")
            
            # Fallback: Load all models from scraped directory
            models = self.storage.load_all_scraped_models()
            self.logger.info(f"Loaded {len(models)} models from scraped files")
        else:
            # Load specific files from scraped directory
            for filename in files:
                try:
                    collection = self.storage.load_models(filename, stage="scraped")
                    if collection:
                        models.extend(collection.models)
                        self.logger.debug(f"Loaded {len(collection.models)} models from {filename}")
                    else:
                        self.logger.warning(f"Could not load data from {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to load {filename}: {e}")
                    continue
        
        # Deduplicate models by name + developer
        unique_models = self._deduplicate_models(models)
        
        # Cache results if loading all files
        if files is None or len(files) > 1:
            self._models_cache = unique_models
            self._cache_timestamp = datetime.utcnow()
        
        self.logger.info(f"Loaded {len(unique_models)} unique models total")
        return unique_models
    
    def _deduplicate_models(self, models: List[Model]) -> List[Model]:
        """Remove duplicate models, preferring higher confidence and more recent data."""
        seen = {}
        
        for model in models:
            key = (model.name.lower(), model.developer.lower())
            
            if key not in seen:
                seen[key] = model
            else:
                existing = seen[key]
                
                # Prefer higher confidence
                if (model.training_flop_confidence.value == "high" and 
                    existing.training_flop_confidence.value != "high"):
                    seen[key] = model
                # If same confidence, prefer more recent
                elif (model.training_flop_confidence == existing.training_flop_confidence and
                      model.last_updated > existing.last_updated):
                    seen[key] = model
        
        return list(seen.values())
    
    def filter_models(
        self, 
        models: List[Model], 
        above_threshold: bool = False,
        developer: Optional[str] = None,
        name: Optional[str] = None,
        min_params: Optional[int] = None,
        confidence: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Model]:
        """Filter models based on criteria.
        
        Args:
            models: List of models to filter
            above_threshold: Only include models above 1e25 FLOP threshold
            developer: Filter by developer name (case insensitive)
            name: Filter by model name (case insensitive substring match)
            min_params: Minimum parameter count
            confidence: Filter by confidence level
            status: Filter by model status
            
        Returns:
            Filtered list of models
        """
        filtered = models.copy()
        
        if above_threshold:
            from ..config.thresholds import ThresholdClassification
            filtered = [
                m for m in filtered 
                if m.get_threshold_classification() in [ThresholdClassification.HIGH_CONFIDENCE_ABOVE, ThresholdClassification.UNCERTAIN]
            ]
        
        if developer:
            dev_lower = developer.lower()
            filtered = [
                m for m in filtered 
                if dev_lower in m.developer.lower() or dev_lower in m.name.lower()
            ]
        
        if name:
            name_lower = name.lower()
            filtered = [
                m for m in filtered 
                if name_lower in m.name.lower()
            ]
        
        if min_params:
            filtered = [
                m for m in filtered 
                if m.parameters and m.parameters >= min_params
            ]
        
        if confidence:
            try:
                conf_level = ConfidenceLevel(confidence.lower())
                filtered = [m for m in filtered if m.training_flop_confidence == conf_level]
            except ValueError:
                self.logger.warning(f"Invalid confidence level: {confidence}")
        
        if status:
            # Note: status filtering deprecated - use threshold_classification instead
            self.logger.warning(f"Status filtering deprecated, ignoring status parameter: {status}")
        
        return filtered
    
    def sort_models(
        self, 
        models: List[Model], 
        by: str = "flop", 
        reverse: bool = True
    ) -> List[Model]:
        """Sort models by specified criteria.
        
        Args:
            models: List of models to sort
            by: Sort criteria ("flop", "params", "date", "name")
            reverse: Sort in descending order (default True)
            
        Returns:
            Sorted list of models
        """
        sort_functions = {
            "flop": lambda m: m.training_flop or 0,
            "params": lambda m: m.parameters or 0,
            "date": lambda m: m.release_date or datetime.min,
            "name": lambda m: m.name.lower(),
            "developer": lambda m: m.developer.lower(),
            "confidence": lambda m: {
                "high": 4, "medium": 3, "low": 2, "speculative": 1
            }.get(m.training_flop_confidence.value, 0)
        }
        
        if by not in sort_functions:
            self.logger.warning(f"Unknown sort criteria: {by}, using 'flop'")
            by = "flop"
        
        try:
            return sorted(models, key=sort_functions[by], reverse=reverse)
        except Exception as e:
            self.logger.error(f"Failed to sort by {by}: {e}")
            return models
    
    def get_top_models(
        self, 
        n: int = 10, 
        files: Optional[List[str]] = None,
        **filter_kwargs
    ) -> List[Model]:
        """Get top N models by training FLOP.
        
        Args:
            n: Number of models to return
            files: Specific data files to load
            **filter_kwargs: Additional filtering criteria
            
        Returns:
            List of top N models
        """
        models = self.load_data(files)
        filtered = self.filter_models(models, **filter_kwargs)
        sorted_models = self.sort_models(filtered, by="flop", reverse=True)
        
        return sorted_models[:n]
    
    def get_available_files(self) -> List[str]:
        """Get list of available data files."""
        return self.storage.list_saved_files(stage="scraped")
    
    def get_available_developers(self) -> List[str]:
        """Get list of available developers."""
        models = self.load_data()
        developers = sorted(set(m.developer for m in models))
        return developers
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset."""
        models = self.load_data()
        
        from ..config.thresholds import ThresholdClassification
        above_threshold = [
            m for m in models 
            if m.get_threshold_classification() in [ThresholdClassification.HIGH_CONFIDENCE_ABOVE, ThresholdClassification.UNCERTAIN]
        ]
        
        confidence_counts = {}
        for conf in ConfidenceLevel:
            confidence_counts[conf.value] = len([
                m for m in models if m.training_flop_confidence == conf
            ])
        
        return {
            "total_models": len(models),
            "above_threshold": len(above_threshold),
            "below_threshold": len(models) - len(above_threshold),
            "with_flop_estimates": len([m for m in models if m.training_flop]),
            "with_parameters": len([m for m in models if m.parameters]),
            "confidence_distribution": confidence_counts,
            "developers": len(set(m.developer for m in models)),
            "data_files": len(self.get_available_files()),
        }
    
    def save_query_results(self, models: List[Model], filename: str) -> Path:
        """Save query results to query_results directory.
        
        Args:
            models: List of models to save
            filename: Filename without extension
            
        Returns:
            Path to saved file
        """
        from datetime import datetime
        
        collection = ModelCollection(
            models=models,
            source=f"query_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return self.storage.save_models(collection, filename, stage="query_results")