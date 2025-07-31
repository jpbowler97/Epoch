#!/usr/bin/env python3
"""Test script for the compute estimation methods."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.estimation import ComputeEstimator


def test_scaling_laws():
    """Test scaling law estimation methods."""
    print("=" * 60)
    print("SCALING LAW ESTIMATION TESTS")
    print("=" * 60)
    
    estimator = ComputeEstimator()
    
    # Test with Llama 3.1 405B (known reference)
    params = 405_000_000_000  # 405B parameters
    tokens = 15_000_000_000_000  # 15T tokens
    
    result = estimator.estimate_from_scaling_laws(params, tokens, "chinchilla")
    print(f"Llama 3.1 405B (Chinchilla): {result.flop_estimate:.2e} FLOP")
    print(f"Confidence: {result.confidence.value}")
    print(f"Reasoning: {result.reasoning}")
    print()
    
    # Test with smaller model
    params_small = 7_000_000_000  # 7B parameters
    tokens_small = 2_000_000_000_000  # 2T tokens
    
    result_small = estimator.estimate_from_scaling_laws(params_small, tokens_small, "chinchilla")
    print(f"Llama 2 7B: {result_small.flop_estimate:.2e} FLOP")
    print(f"Above 1e25 threshold: {result_small.flop_estimate >= 1e25}")
    print()


def test_hardware_estimation():
    """Test hardware-based estimation."""
    print("=" * 60)
    print("HARDWARE-BASED ESTIMATION TESTS")
    print("=" * 60)
    
    estimator = ComputeEstimator()
    
    # Test with Llama 3.1 405B known specs
    result = estimator.estimate_from_hardware(
        gpu_type="H100",
        gpu_count=16000,
        training_time_hours=2000,  # Estimated
        utilization=0.6,
        efficiency_scenario="large_scale"
    )
    
    print(f"Llama 3.1 405B (Hardware): {result.flop_estimate:.2e} FLOP")
    print(f"Confidence: {result.confidence.value}")
    print(f"Reasoning: {result.reasoning}")
    print()
    
    # Test with smaller setup
    result_small = estimator.estimate_from_hardware(
        gpu_type="A100",
        gpu_count=1024,
        training_time_hours=500,
        utilization=0.7,
        efficiency_scenario="multi_node"
    )
    
    print(f"Mid-size model (1k A100s): {result_small.flop_estimate:.2e} FLOP")
    print(f"Above 1e25 threshold: {result_small.flop_estimate >= 1e25}")
    print()


def test_benchmark_estimation():
    """Test benchmark-based estimation."""
    print("=" * 60)
    print("BENCHMARK-BASED ESTIMATION TESTS")
    print("=" * 60)
    
    estimator = ComputeEstimator()
    
    # Test with high-performing model
    result = estimator.estimate_from_benchmark_elo(
        elo_rating=1400,  # High-performing model
        reference_model="Llama 3.1 405B",
        reference_flop=3.8e25,
        reference_elo=1350
    )
    
    print(f"High ELO model (1400): {result.flop_estimate:.2e} FLOP")
    print(f"Confidence: {result.confidence.value}")
    print(f"Reasoning: {result.reasoning}")
    print()
    
    # Test with lower-performing model
    result_low = estimator.estimate_from_benchmark_elo(
        elo_rating=1200,
        reference_model="Llama 3.1 405B", 
        reference_flop=3.8e25,
        reference_elo=1350
    )
    
    print(f"Lower ELO model (1200): {result_low.flop_estimate:.2e} FLOP")
    print(f"Above 1e25 threshold: {result_low.flop_estimate >= 1e25}")
    print()


def test_cost_estimation():
    """Test cost-based estimation."""
    print("=" * 60)
    print("COST-BASED ESTIMATION TESTS")
    print("=" * 60)
    
    estimator = ComputeEstimator()
    
    # Test with GPT-4 reported cost
    result = estimator.estimate_from_cost_disclosure(
        training_cost_usd=78_000_000,  # $78M reported for GPT-4
        gpu_type="H100",
        cloud_provider="aws",
        year=2024
    )
    
    print(f"GPT-4 ($78M cost): {result.flop_estimate:.2e} FLOP")
    print(f"Confidence: {result.confidence.value}")
    print(f"Reasoning: {result.reasoning}")
    print()


def test_best_estimate():
    """Test the best estimate method with multiple inputs."""
    print("=" * 60)
    print("BEST ESTIMATE TEST (Multiple Methods)")
    print("=" * 60)
    
    estimator = ComputeEstimator()
    
    # Test with comprehensive data (simulate Meta Llama 3.1 405B)
    result = estimator.get_best_estimate(
        parameters=405_000_000_000,
        tokens=15_000_000_000_000,
        gpu_type="H100",
        gpu_count=16000,
        training_hours=2000,
        utilization=0.6,
        elo_rating=1350,
        training_cost_usd=100_000_000  # Estimated
    )
    
    if result:
        print(f"Best estimate: {result.flop_estimate:.2e} FLOP")
        print(f"Method: {result.method.value}")
        print(f"Confidence: {result.confidence.value}")
        print(f"Reasoning: {result.reasoning}")
        print()
        print(f"Above 1e25 threshold: {result.flop_estimate >= 1e25}")
    else:
        print("No valid estimate found")


if __name__ == "__main__":
    test_scaling_laws()
    test_hardware_estimation()
    test_benchmark_estimation()
    test_cost_estimation()
    test_best_estimate()