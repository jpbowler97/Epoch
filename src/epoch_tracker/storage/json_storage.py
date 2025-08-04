"""JSON-based storage for model data."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..models import Model, ModelCollection


# Simple FLOP formatting functions
def format_flop_value(flop: Optional[float]) -> Optional[str]:
    """Format FLOP value for JSON serialization."""
    if flop is None:
        return None
    return f"{flop:.2e}"


def parse_flop_value(flop_str: Optional[str]) -> Optional[float]:
    """Parse FLOP value from JSON."""
    if flop_str is None:
        return None
    try:
        return float(flop_str)
    except (ValueError, TypeError):
        return None


class JSONStorage:
    """Handle saving and loading model data in JSON format."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize JSON storage.
        
        Args:
            data_dir: Base data directory (uses config default if None)
        """
        if data_dir is None:
            data_dir = get_settings().data_dir
        
        self.data_dir = Path(data_dir)
        self.scraped_dir = self.data_dir / "scraped"
        self.processed_dir = self.data_dir / "processed"  # Legacy support
        self.estimated_dir = self.data_dir / "estimated"
        self.query_results_dir = self.data_dir / "query_results"
        
        # Create all directories
        self.scraped_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)  # Legacy support
        self.estimated_dir.mkdir(parents=True, exist_ok=True)
        self.query_results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_models(self, collection: ModelCollection, filename: str, stage: str = "processed") -> Path:
        """Save a model collection to JSON file.
        
        Args:
            collection: ModelCollection to save
            filename: Filename without extension
            stage: Directory stage - "scraped", "processed", "estimated", or "query_results" (default: "processed")
            
        Returns:
            Path to saved file
        """
        if stage == "scraped":
            target_dir = self.scraped_dir
        elif stage == "processed":
            target_dir = self.processed_dir
        elif stage == "estimated":
            target_dir = self.estimated_dir
        elif stage == "query_results":
            target_dir = self.query_results_dir
        else:
            raise ValueError(f"Invalid stage '{stage}'. Must be 'scraped', 'processed', 'estimated', or 'query_results'")
            
        filepath = target_dir / f"{filename}.json"
        
        # Convert to dict format for JSON serialization
        data = {
            "metadata": {
                "saved_at": datetime.utcnow().isoformat(),
                "source": collection.source,
                "last_updated": collection.last_updated.isoformat(),
                "model_count": len(collection.models),
                "stage": stage
            },
            "models": [self._model_to_dict(model) for model in collection.models]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def load_models(self, filename: str, stage: str = "processed") -> Optional[ModelCollection]:
        """Load a model collection from JSON file.
        
        Args:
            filename: Filename without extension
            stage: Directory stage - "scraped", "processed", "estimated", or "query_results" (default: "processed")
            
        Returns:
            ModelCollection or None if file not found
        """
        if stage == "scraped":
            target_dir = self.scraped_dir
        elif stage == "processed":
            target_dir = self.processed_dir
        elif stage == "estimated":
            target_dir = self.estimated_dir
        elif stage == "query_results":
            target_dir = self.query_results_dir
        else:
            raise ValueError(f"Invalid stage '{stage}'. Must be 'scraped', 'processed', 'estimated', or 'query_results'")
            
        filepath = target_dir / f"{filename}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to model objects
            models = [self._dict_to_model(model_data) for model_data in data["models"]]
            
            metadata = data.get("metadata", {})
            collection = ModelCollection(
                models=models,
                source=metadata.get("source", "unknown"),
            )
            
            # Set last_updated if available
            if "last_updated" in metadata:
                from ..utils import parse_date
                collection.last_updated = parse_date(metadata["last_updated"])
            
            return collection
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            raise ValueError(f"Failed to load models from {filepath}: {e}")
    
    def list_saved_files(self, stage: str = "processed") -> List[str]:
        """List all saved model files.
        
        Args:
            stage: Directory stage - "scraped", "processed", "estimated", or "query_results" (default: "processed")
        
        Returns:
            List of filenames (without extensions)
        """
        if stage == "scraped":
            target_dir = self.scraped_dir
        elif stage == "processed":
            target_dir = self.processed_dir
        elif stage == "estimated":
            target_dir = self.estimated_dir
        elif stage == "query_results":
            target_dir = self.query_results_dir
        else:
            raise ValueError(f"Invalid stage '{stage}'. Must be 'scraped', 'processed', 'estimated', or 'query_results'")
            
        json_files = target_dir.glob("*.json")
        return [f.stem for f in json_files]
    
    def load_all_scraped_models(self) -> List[Model]:
        """Load all models from all scraped files.
        
        Returns:
            List of all models from scraped directory
        """
        all_models = []
        scraped_files = self.list_saved_files(stage="scraped")
        
        for filename in scraped_files:
            collection = self.load_models(filename, stage="scraped")
            if collection:
                all_models.extend(collection.models)
        
        return all_models
    
    def export_csv(self, collection: ModelCollection, filename: str) -> Path:
        """Export model collection to CSV format.
        
        Args:
            collection: ModelCollection to export
            filename: Filename without extension
            
        Returns:
            Path to CSV file
        """
        import pandas as pd
        
        # Convert models to DataFrame
        rows = []
        for model in collection.models:
            row = {
                "name": model.name,
                "developer": model.developer,
                "release_date": model.release_date.isoformat() if model.release_date else None,
                "parameters": model.parameters,
                "training_flop": model.training_flop,
                "training_flop_confidence": model.training_flop_confidence.value,
                "estimation_method": model.estimation_method.value,
                "status": model.status.value,
                "reasoning": model.reasoning,
                "sources": "; ".join(model.sources),
                "benchmarks": json.dumps(model.benchmarks) if model.benchmarks else None,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        csv_path = self.query_results_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def _model_to_dict(self, model: Model) -> Dict[str, Any]:
        """Convert Model to dictionary for JSON serialization."""
        return {
            "name": model.name,
            "developer": model.developer,
            "release_date": model.release_date.isoformat() if model.release_date else None,
            "parameters": model.parameters,
            "parameter_source": model.parameter_source,
            "context_length": model.context_length,
            "architecture": model.architecture,
            "training_flop": format_flop_value(model.training_flop),
            "training_flop_confidence": model.training_flop_confidence.value,
            "estimation_method": model.estimation_method.value,
            "alternative_estimates": [
                {
                    "flop": format_flop_value(est.flop),
                    "confidence": est.confidence.value,
                    "method": est.method.value,
                    "reasoning": est.reasoning
                } for est in model.alternative_estimates
            ],
            "inference_flop_per_token": format_flop_value(model.inference_flop_per_token),
            "status": model.status.value,
            "benchmarks": model.benchmarks,
            "sources": model.sources,
            "reasoning": model.reasoning,
            "last_updated": model.last_updated.isoformat(),
            "metadata": model.metadata,
        }
    
    def _dict_to_model(self, data: Dict[str, Any]) -> Model:
        """Convert dictionary to Model object."""
        from ..models import ConfidenceLevel, EstimationMethod, ModelStatus, AlternativeEstimate
        from ..utils import parse_date
        
        # Parse alternative estimates
        alternative_estimates = []
        for est_data in data.get("alternative_estimates", []):
            alternative_estimates.append(AlternativeEstimate(
                flop=parse_flop_value(est_data.get("flop")),
                confidence=ConfidenceLevel(est_data.get("confidence", "speculative")),
                method=EstimationMethod(est_data.get("method", "manual_research")),
                reasoning=est_data.get("reasoning", "")
            ))
        
        return Model(
            name=data["name"],
            developer=data["developer"],
            release_date=parse_date(data.get("release_date")),
            parameters=data.get("parameters"),
            parameter_source=data.get("parameter_source"),
            context_length=data.get("context_length"),
            architecture=data.get("architecture"),
            training_flop=parse_flop_value(data.get("training_flop")),
            training_flop_confidence=ConfidenceLevel(data.get("training_flop_confidence", "speculative")),
            estimation_method=EstimationMethod(data.get("estimation_method", "manual_research")),
            alternative_estimates=alternative_estimates,
            inference_flop_per_token=parse_flop_value(data.get("inference_flop_per_token")),
            status=ModelStatus(data.get("status", "uncertain")),
            benchmarks=data.get("benchmarks", {}),
            sources=data.get("sources", []),
            reasoning=data.get("reasoning", ""),
            last_updated=parse_date(data.get("last_updated")) or datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )