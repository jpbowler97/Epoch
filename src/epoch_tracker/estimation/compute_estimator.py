"""Compute estimation methods for AI model training FLOP calculations."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
from enum import Enum

from ..models import ConfidenceLevel, EstimationMethod


class GPUType(Enum):
    """GPU types with their FLOP specifications."""
    H100 = "H100"
    A100 = "A100"
    V100 = "V100"
    TPU_V4 = "TPU_V4"
    TPU_V5 = "TPU_V5"


@dataclass
class GPUSpecs:
    """Hardware specifications for different GPU types."""
    peak_flops_bf16: float  # Peak FLOP/s for BF16
    memory_gb: int
    tensor_cores: bool = True


@dataclass
class EstimationResult:
    """Result of a FLOP estimation calculation."""
    flop_estimate: float
    confidence: ConfidenceLevel
    method: EstimationMethod
    reasoning: str
    metadata: Dict[str, Any]


class ComputeEstimator:
    """Advanced compute estimation using multiple methodologies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # GPU specifications (peak BF16/FP16 performance)
        self.gpu_specs = {
            GPUType.H100: GPUSpecs(peak_flops_bf16=2e15, memory_gb=80),
            GPUType.A100: GPUSpecs(peak_flops_bf16=3.1e14, memory_gb=80),
            GPUType.V100: GPUSpecs(peak_flops_bf16=1.25e14, memory_gb=32),
            GPUType.TPU_V4: GPUSpecs(peak_flops_bf16=1.2e15, memory_gb=32),
            GPUType.TPU_V5: GPUSpecs(peak_flops_bf16=1.9e15, memory_gb=96),
        }
        
        # Efficiency factors for different training setups
        self.efficiency_factors = {
            "single_node": 0.9,     # Single machine efficiency
            "multi_node": 0.7,      # Multi-node distributed training
            "large_scale": 0.5,     # Very large scale (>10k GPUs)
        }

    def estimate_from_scaling_laws(
        self, 
        parameters: int, 
        tokens: int, 
        method: str = "chinchilla"
    ) -> EstimationResult:
        """Estimate FLOP using scaling laws.
        
        Args:
            parameters: Number of model parameters
            tokens: Number of training tokens
            method: Scaling law method ("chinchilla", "kaplan", "basic")
        """
        # Validate inputs
        if parameters is None or tokens is None:
            raise ValueError(f"Both parameters ({parameters}) and tokens ({tokens}) must be provided for scaling law estimation")
        
        if parameters <= 0 or tokens <= 0:
            raise ValueError(f"Parameters ({parameters}) and tokens ({tokens}) must be positive")
        
        if method == "chinchilla":
            # Chinchilla scaling: 6 * N * D (most accurate)
            flop_estimate = 6 * parameters * tokens
            confidence = ConfidenceLevel.HIGH
            reasoning = f"Chinchilla scaling law: 6 × {parameters:,} params × {tokens:,} tokens = {flop_estimate:.2e} FLOP"
            
        elif method == "kaplan":
            # Kaplan scaling: ~6 * N * D (similar but different assumptions)
            flop_estimate = 5.5 * parameters * tokens
            confidence = ConfidenceLevel.MEDIUM
            reasoning = f"Kaplan scaling law: 5.5 × {parameters:,} params × {tokens:,} tokens = {flop_estimate:.2e} FLOP"
            
        else:  # basic
            # Simple approximation: 2 * N * D (forward + backward pass)
            flop_estimate = 2 * parameters * tokens
            confidence = ConfidenceLevel.LOW
            reasoning = f"Basic scaling law: 2 × {parameters:,} params × {tokens:,} tokens = {flop_estimate:.2e} FLOP"
        
        return EstimationResult(
            flop_estimate=flop_estimate,
            confidence=confidence,
            method=EstimationMethod.SCALING_LAWS,
            reasoning=reasoning,
            metadata={
                "parameters": parameters,
                "tokens": tokens,
                "scaling_method": method
            }
        )

    def estimate_from_hardware(
        self,
        gpu_type: str,
        gpu_count: int,
        training_time_hours: float,
        utilization: float = 0.5,
        efficiency_scenario: str = "multi_node"
    ) -> EstimationResult:
        """Estimate FLOP from hardware specifications.
        
        Args:
            gpu_type: Type of GPU/accelerator
            gpu_count: Number of GPUs
            training_time_hours: Total training time in hours
            utilization: GPU utilization factor (0.0-1.0)
            efficiency_scenario: Training efficiency scenario
        """
        try:
            # Parse GPU type
            gpu_enum = None
            for gpu in GPUType:
                if gpu.value.upper() in gpu_type.upper():
                    gpu_enum = gpu
                    break
            
            if not gpu_enum:
                raise ValueError(f"Unknown GPU type: {gpu_type}")
            
            gpu_spec = self.gpu_specs[gpu_enum]
            efficiency = self.efficiency_factors[efficiency_scenario]
            
            # Calculate total FLOP
            peak_flops_total = gpu_spec.peak_flops_bf16 * gpu_count
            training_seconds = training_time_hours * 3600
            effective_flops = peak_flops_total * utilization * efficiency
            total_flop = effective_flops * training_seconds
            
            # Determine confidence based on information quality
            if utilization == 0.5 and efficiency_scenario == "multi_node":
                confidence = ConfidenceLevel.MEDIUM  # Using default assumptions
            else:
                confidence = ConfidenceLevel.HIGH  # Have specific utilization data
            
            reasoning = (f"Hardware-based calculation: {gpu_count:,} × {gpu_type} GPUs "
                        f"for {training_time_hours:,.0f}h at {utilization:.1%} utilization "
                        f"with {efficiency:.1%} efficiency = {total_flop:.2e} FLOP")
            
            return EstimationResult(
                flop_estimate=total_flop,
                confidence=confidence,
                method=EstimationMethod.DIRECT_CALCULATION,
                reasoning=reasoning,
                metadata={
                    "gpu_type": gpu_type,
                    "gpu_count": gpu_count,
                    "training_hours": training_time_hours,
                    "utilization": utilization,
                    "efficiency_scenario": efficiency_scenario,
                    "peak_flops_per_gpu": gpu_spec.peak_flops_bf16
                }
            )
            
        except Exception as e:
            self.logger.error(f"Hardware estimation failed: {e}")
            raise ValueError(f"Cannot estimate FLOP from hardware: {e}")

    def estimate_from_benchmark_elo(
        self,
        elo_rating: float,
        reference_model: str = "llama_405b",
        reference_flop: float = 3.8e25,
        reference_elo: float = 1350
    ) -> EstimationResult:
        """Estimate FLOP using ELO rating interpolation.
        
        Args:
            elo_rating: ChatBot Arena ELO rating
            reference_model: Reference model for scaling
            reference_flop: Known FLOP of reference model
            reference_elo: ELO rating of reference model
        """
        try:
            # Use power law scaling between ELO and compute
            # Assumption: FLOP ∝ ELO^α where α ≈ 2-4 based on scaling laws
            alpha = 3.0  # Conservative middle estimate
            
            flop_ratio = (elo_rating / reference_elo) ** alpha
            flop_estimate = reference_flop * flop_ratio
            
            # Confidence decreases with distance from reference point
            elo_diff = abs(elo_rating - reference_elo)
            if elo_diff < 50:
                confidence = ConfidenceLevel.MEDIUM
            elif elo_diff < 100:
                confidence = ConfidenceLevel.LOW
            else:
                confidence = ConfidenceLevel.SPECULATIVE
            
            reasoning = (f"Benchmark-based estimation: ELO {elo_rating} vs reference "
                        f"{reference_model} (ELO {reference_elo}, {reference_flop:.2e} FLOP) "
                        f"with power law scaling α={alpha} = {flop_estimate:.2e} FLOP")
            
            return EstimationResult(
                flop_estimate=flop_estimate,
                confidence=confidence,
                method=EstimationMethod.BENCHMARK_BASED,
                reasoning=reasoning,
                metadata={
                    "elo_rating": elo_rating,
                    "reference_model": reference_model,
                    "reference_flop": reference_flop,
                    "reference_elo": reference_elo,
                    "scaling_exponent": alpha
                }
            )
            
        except Exception as e:
            self.logger.error(f"Benchmark estimation failed: {e}")
            raise ValueError(f"Cannot estimate FLOP from ELO: {e}")

    def estimate_from_cost_disclosure(
        self,
        training_cost_usd: float,
        gpu_type: str = "H100",
        cloud_provider: str = "aws",
        year: int = 2024
    ) -> EstimationResult:
        """Estimate FLOP from disclosed training costs.
        
        Args:
            training_cost_usd: Training cost in USD
            gpu_type: Type of GPU used
            cloud_provider: Cloud provider for pricing
            year: Year for pricing reference
        """
        try:
            # Cloud GPU pricing (USD per hour, approximate 2024 rates)
            gpu_pricing = {
                "H100": {"aws": 30, "gcp": 28, "azure": 32},
                "A100": {"aws": 10, "gcp": 9, "azure": 11},
                "V100": {"aws": 3, "gcp": 2.5, "azure": 3.5},
            }
            
            if gpu_type not in gpu_pricing:
                raise ValueError(f"No pricing data for GPU type: {gpu_type}")
            
            price_per_hour = gpu_pricing[gpu_type].get(cloud_provider, 
                                                    list(gpu_pricing[gpu_type].values())[0])
            
            # Calculate total GPU hours from cost
            total_gpu_hours = training_cost_usd / price_per_hour
            
            # Convert to FLOP using hardware specs (assume 50% utilization)
            gpu_enum = None
            for gpu in GPUType:
                if gpu.value.upper() in gpu_type.upper():
                    gpu_enum = gpu
                    break
            
            if not gpu_enum:
                raise ValueError(f"Unknown GPU type: {gpu_type}")
            
            gpu_spec = self.gpu_specs[gpu_enum]
            utilization = 0.5  # Conservative assumption
            efficiency = 0.6   # Cloud efficiency
            
            effective_flops_per_hour = gpu_spec.peak_flops_bf16 * utilization * efficiency * 3600
            total_flop = total_gpu_hours * effective_flops_per_hour
            
            confidence = ConfidenceLevel.LOW  # Many assumptions in cost-based estimation
            
            reasoning = (f"Cost-based estimation: ${training_cost_usd:,} at ${price_per_hour}/h "
                        f"for {gpu_type} = {total_gpu_hours:,.0f} GPU-hours "
                        f"at {utilization:.1%} utilization = {total_flop:.2e} FLOP")
            
            return EstimationResult(
                flop_estimate=total_flop,
                confidence=confidence,
                method=EstimationMethod.DIRECT_CALCULATION,
                reasoning=reasoning,
                metadata={
                    "training_cost_usd": training_cost_usd,
                    "gpu_type": gpu_type,
                    "cloud_provider": cloud_provider,
                    "price_per_hour": price_per_hour,
                    "total_gpu_hours": total_gpu_hours,
                    "year": year
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cost-based estimation failed: {e}")
            raise ValueError(f"Cannot estimate FLOP from cost: {e}")

    def get_best_estimate(self, **kwargs) -> Optional[EstimationResult]:
        """Get the best available FLOP estimate from multiple methods.
        
        Tries multiple estimation methods and returns the highest confidence result.
        """
        estimates = []
        
        # Try scaling laws if we have parameters and tokens
        if kwargs.get("parameters") and kwargs.get("tokens"):
            try:
                estimate = self.estimate_from_scaling_laws(
                    kwargs["parameters"], 
                    kwargs["tokens"],
                    kwargs.get("scaling_method", "chinchilla")
                )
                estimates.append(estimate)
            except Exception as e:
                self.logger.debug(f"Scaling law estimation failed: {e}")
        
        # Try hardware-based if we have hardware specs
        if kwargs.get("gpu_type") and kwargs.get("gpu_count") and kwargs.get("training_hours"):
            try:
                estimate = self.estimate_from_hardware(
                    kwargs["gpu_type"],
                    kwargs["gpu_count"],
                    kwargs["training_hours"],
                    kwargs.get("utilization", 0.5),
                    kwargs.get("efficiency_scenario", "multi_node")
                )
                estimates.append(estimate)
            except Exception as e:
                self.logger.debug(f"Hardware estimation failed: {e}")
        
        # Try benchmark-based if we have ELO rating
        if kwargs.get("elo_rating"):
            try:
                estimate = self.estimate_from_benchmark_elo(
                    kwargs["elo_rating"],
                    kwargs.get("reference_model", "llama_405b"),
                    kwargs.get("reference_flop", 3.8e25),
                    kwargs.get("reference_elo", 1350)
                )
                estimates.append(estimate)
            except Exception as e:
                self.logger.debug(f"Benchmark estimation failed: {e}")
        
        # Try cost-based if we have training cost
        if kwargs.get("training_cost_usd"):
            try:
                estimate = self.estimate_from_cost_disclosure(
                    kwargs["training_cost_usd"],
                    kwargs.get("gpu_type", "H100"),
                    kwargs.get("cloud_provider", "aws"),
                    kwargs.get("year", 2024)
                )
                estimates.append(estimate)
            except Exception as e:
                self.logger.debug(f"Cost estimation failed: {e}")
        
        if not estimates:
            return None
        
        # Return the estimate with highest confidence
        confidence_order = [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, 
                          ConfidenceLevel.LOW, ConfidenceLevel.SPECULATIVE]
        
        for confidence in confidence_order:
            for estimate in estimates:
                if estimate.confidence == confidence:
                    return estimate
        
        return estimates[0]  # Fallback to first estimate