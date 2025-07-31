"""Core data models for AI model tracking."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator


class ConfidenceLevel(Enum):
    """Confidence levels for FLOP estimates."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


class EstimationMethod(Enum):
    """Methods used to estimate training FLOP."""
    DIRECT_CALCULATION = "direct_calculation"
    BENCHMARK_BASED = "benchmark_based"
    SCALING_LAWS = "scaling_laws"
    MANUAL_RESEARCH = "manual_research"
    COMPANY_DISCLOSURE = "company_disclosure"
    EPOCH_ESTIMATE = "epoch_estimate"


class ModelStatus(Enum):
    """Status of model in tracking system."""
    CONFIRMED_ABOVE = "confirmed_above_1e25"
    LIKELY_ABOVE = "likely_above_1e25"
    UNCERTAIN = "uncertain"
    LIKELY_BELOW = "likely_below_1e25"
    CONFIRMED_BELOW = "confirmed_below_1e25"


class ThresholdClassification(Enum):
    """Threshold-based classification for FLOP estimates."""
    HIGH_CONFIDENCE_ABOVE = "high_confidence_above_1e25"
    HIGH_CONFIDENCE_BELOW = "high_confidence_below_1e25"
    NOT_SURE = "not_sure"


class AlternativeEstimate(BaseModel):
    """Alternative FLOP estimate using a different method."""
    flop: float = Field(..., description="Training FLOP estimate")
    confidence: ConfidenceLevel = Field(..., description="Confidence in this estimate")
    method: EstimationMethod = Field(..., description="Method used for this estimate")
    reasoning: str = Field("", description="Explanation for this estimate")
    
    @validator('flop')
    def validate_flop(cls, v):
        """Ensure FLOP is positive."""
        if v <= 0:
            raise ValueError("FLOP must be positive")
        return v


class Model(BaseModel):
    """Core model representing an AI system with training compute estimates."""
    
    # Basic identification
    name: str = Field(..., description="Model name")
    developer: str = Field(..., description="Organization that developed the model")
    release_date: Optional[datetime] = Field(None, description="When the model was released")
    
    # Technical specifications
    parameters: Optional[int] = Field(None, description="Number of parameters")
    parameter_source: Optional[str] = Field(None, description="Source of parameter count (e.g., 'extracted_from_name', 'official_disclosure')")
    context_length: Optional[int] = Field(None, description="Maximum context window")
    architecture: Optional[str] = Field(None, description="Model architecture (e.g., transformer)")
    
    # Training compute estimates (primary)
    training_flop: Optional[float] = Field(None, description="Primary training FLOP estimate")
    training_flop_confidence: ConfidenceLevel = Field(
        ConfidenceLevel.SPECULATIVE, description="Confidence in primary FLOP estimate"
    )
    estimation_method: EstimationMethod = Field(
        EstimationMethod.MANUAL_RESEARCH, description="Primary method used for FLOP estimation"
    )
    
    # Alternative estimates
    alternative_estimates: List[AlternativeEstimate] = Field(
        default_factory=list, description="Alternative FLOP estimates using different methods"
    )
    
    # Inference compute (bonus metric)
    inference_flop_per_token: Optional[float] = Field(
        None, description="FLOP required per output token"
    )
    
    # Classification
    status: ModelStatus = Field(
        ModelStatus.UNCERTAIN, description="Classification relative to 1e25 FLOP threshold"
    )
    threshold_classification: ThresholdClassification = Field(
        ThresholdClassification.NOT_SURE, description="Threshold-based classification for FLOP estimates"
    )
    
    # Supporting data
    benchmarks: Dict[str, float] = Field(
        default_factory=dict, description="Benchmark scores (name -> score)"
    )
    sources: List[str] = Field(
        default_factory=list, description="URLs and references for data"
    )
    reasoning: str = Field("", description="Explanation for classification and estimates")
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata and raw data"
    )
    
    @validator('training_flop')
    def validate_training_flop(cls, v):
        """Ensure training FLOP is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Training FLOP must be positive")
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v):
        """Ensure parameter count is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Parameter count must be positive")
        return v
    
    def is_above_threshold(self, threshold: float = 1e25) -> Optional[bool]:
        """Check if model is above the FLOP threshold.
        
        Returns:
            True if confirmed above, False if confirmed below, None if uncertain
        """
        if self.training_flop is None:
            return None
        
        if self.training_flop_confidence == ConfidenceLevel.HIGH:
            return self.training_flop >= threshold
        elif self.training_flop_confidence == ConfidenceLevel.MEDIUM:
            # Use 50% margin for medium confidence
            return self.training_flop >= threshold * 0.5
        else:
            # Low/speculative confidence - return None (uncertain)
            return None
    
    def classify_by_threshold(
        self, 
        high_confidence_above_threshold: float = 1e26,
        high_confidence_below_threshold: float = 5e24
    ) -> ThresholdClassification:
        """Classify model based on FLOP estimate and confidence thresholds.
        
        Args:
            high_confidence_above_threshold: FLOP threshold for high confidence above 1e25
            high_confidence_below_threshold: FLOP threshold for high confidence below 1e25
            
        Returns:
            ThresholdClassification based on FLOP estimate
        """
        if self.training_flop is None:
            return ThresholdClassification.NOT_SURE
        
        if self.training_flop >= high_confidence_above_threshold:
            return ThresholdClassification.HIGH_CONFIDENCE_ABOVE
        elif self.training_flop <= high_confidence_below_threshold:
            return ThresholdClassification.HIGH_CONFIDENCE_BELOW
        else:
            return ThresholdClassification.NOT_SURE
    
    def add_source(self, url: str, description: str = ""):
        """Add a source URL with optional description."""
        if description:
            source_entry = f"{url} ({description})"
        else:
            source_entry = url
        
        if source_entry not in self.sources:
            self.sources.append(source_entry)
    
    def add_benchmark(self, name: str, score: float):
        """Add a benchmark score."""
        self.benchmarks[name] = score
    
    def add_alternative_estimate(
        self, 
        flop: float, 
        confidence: ConfidenceLevel, 
        method: EstimationMethod, 
        reasoning: str = ""
    ):
        """Add an alternative FLOP estimate."""
        estimate = AlternativeEstimate(
            flop=flop,
            confidence=confidence,
            method=method,
            reasoning=reasoning
        )
        self.alternative_estimates.append(estimate)
    
    def update_flop_estimate(
        self, 
        flop: float, 
        confidence: ConfidenceLevel, 
        method: EstimationMethod,
        reasoning: str = "",
        high_confidence_above_threshold: float = 1e26,
        high_confidence_below_threshold: float = 5e24
    ):
        """Update the FLOP estimate with new information."""
        self.training_flop = flop
        self.training_flop_confidence = confidence
        self.estimation_method = method
        if reasoning:
            self.reasoning = reasoning
        self.last_updated = datetime.utcnow()
        
        # Update status based on new estimate
        if self.is_above_threshold() is True:
            if confidence == ConfidenceLevel.HIGH:
                self.status = ModelStatus.CONFIRMED_ABOVE
            else:
                self.status = ModelStatus.LIKELY_ABOVE
        elif self.is_above_threshold() is False:
            if confidence == ConfidenceLevel.HIGH:
                self.status = ModelStatus.CONFIRMED_BELOW
            else:
                self.status = ModelStatus.LIKELY_BELOW
        else:
            self.status = ModelStatus.UNCERTAIN
        
        # Update threshold classification
        self.threshold_classification = self.classify_by_threshold(
            high_confidence_above_threshold, high_confidence_below_threshold
        )


class ModelCollection(BaseModel):
    """Collection of models with metadata."""
    
    models: List[Model] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field("", description="Source of the model data")
    
    def add_model(self, model: Model):
        """Add a model to the collection."""
        self.models.append(model)
        self.last_updated = datetime.utcnow()
    
    def get_models_above_threshold(self, threshold: float = 1e25) -> List[Model]:
        """Get models that are likely or confirmed above the threshold."""
        return [
            model for model in self.models
            if model.status in [ModelStatus.CONFIRMED_ABOVE, ModelStatus.LIKELY_ABOVE]
        ]
    
    def get_by_threshold_classification(self, classification: ThresholdClassification) -> List[Model]:
        """Get models by threshold classification."""
        return [
            model for model in self.models
            if model.threshold_classification == classification
        ]
    
    def get_high_confidence_above_models(self) -> List[Model]:
        """Get models with high confidence above 1e25 FLOP."""
        return self.get_by_threshold_classification(ThresholdClassification.HIGH_CONFIDENCE_ABOVE)
    
    def get_high_confidence_below_models(self) -> List[Model]:
        """Get models with high confidence below 1e25 FLOP."""
        return self.get_by_threshold_classification(ThresholdClassification.HIGH_CONFIDENCE_BELOW)
    
    def get_not_sure_models(self) -> List[Model]:
        """Get models with uncertain threshold classification."""
        return self.get_by_threshold_classification(ThresholdClassification.NOT_SURE)
    
    def get_by_developer(self, developer: str) -> List[Model]:
        """Get all models from a specific developer."""
        return [model for model in self.models if model.developer.lower() == developer.lower()]