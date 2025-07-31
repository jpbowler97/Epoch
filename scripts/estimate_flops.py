#!/usr/bin/env python3
"""
Apply FLOP estimations to scraped models.

This script processes models from the data store and estimates their training FLOP
using various methods including benchmark-based estimation, known model mappings,
and heuristics based on model names and metadata.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.epoch_tracker.estimation import ComputeEstimator
from src.epoch_tracker.models import Model, ModelCollection, ConfidenceLevel, EstimationMethod, ThresholdClassification
from src.epoch_tracker.storage import JSONStorage


# Threshold classification constants
HIGH_CONFIDENCE_ABOVE_THRESHOLD = 5e25  # >= X FLOP = high confidence above 1e25
HIGH_CONFIDENCE_BELOW_THRESHOLD = 9e24  # <= X FLOP = high confidence below 1e25

# Manual overrides based on Epoch AI's tracker (https://epoch.ai/data-insights/models-over-1e25-flop)
# These take HIGHEST PRIORITY and override all other estimation methods
# NOTE: Use exact FLOP values from Epoch's research with appropriate confidence levels
MANUAL_OVERRIDES = {
    # Frontier models from Epoch AI tracker - High precision estimates
    "llama_3.1_405b": (3.8e25, ConfidenceLevel.HIGH, "Epoch AI: High-precision estimate from Meta disclosure"),
    "grok_2": (3.0e25, ConfidenceLevel.HIGH, "Epoch AI: High-precision estimate from xAI disclosure"),
    
    # Low-precision but reliable estimates from Epoch AI
    "claude_3_opus": (1.6e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from industry analysis"),
    "claude_opus": (1.6e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from industry analysis"),  # Alias
    "gpt_4": (2.1e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from scaling analysis"),
    "gemini_1.0_ultra": (5.0e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from Google hints"),
    "gemini_ultra": (5.0e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from Google hints"),  # Alias
    "claude_3.5_sonnet": (3.6e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from benchmarks"),
    "gpt_4o": (3.8e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from OpenAI patterns"),
    
    # Speculative but notable estimates from Epoch AI
    "claude_opus_4": (1.5e26, ConfidenceLevel.LOW, "Epoch AI: Speculative estimate for next-gen Claude"),
    "claude_4_opus": (1.5e26, ConfidenceLevel.LOW, "Epoch AI: Speculative estimate for next-gen Claude"),  # Alias
    "gpt_4.5": (6.4e25, ConfidenceLevel.LOW, "Epoch AI: Low-precision estimate for GPT-4.5"),
    "gpt_4.5_preview": (6.4e25, ConfidenceLevel.LOW, "Epoch AI: Low-precision estimate for GPT-4.5"),  # Alias

    # More manual overrides
    "deepseek_r1": (3.0e24, ConfidenceLevel.HIGH, "https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1"),
}

# Known model specifications for direct FLOP calculation
# NOTE: Patterns must match normalized model names (underscores, not hyphens)
# To add new models: Include normalized name pattern, parameters, training tokens, confidence level
KNOWN_MODEL_SPECS = {
    # Llama models (Meta) - HIGH confidence from official disclosures
    "llama_3.1_405b": (405_000_000_000, 15_000_000_000_000, ConfidenceLevel.HIGH),
    "llama_3.1_70b": (70_000_000_000, 15_000_000_000_000, ConfidenceLevel.HIGH),
    "llama_3.1_8b": (8_000_000_000, 15_000_000_000_000, ConfidenceLevel.HIGH),
    "llama_3_405b": (405_000_000_000, 15_000_000_000_000, ConfidenceLevel.HIGH),
    "llama_3_70b": (70_000_000_000, 15_000_000_000_000, ConfidenceLevel.HIGH),
    "llama_3_8b": (8_000_000_000, 15_000_000_000_000, ConfidenceLevel.HIGH),
    "llama_2_70b": (70_000_000_000, 2_000_000_000_000, ConfidenceLevel.HIGH),
    "llama_2_7b": (7_000_000_000, 2_000_000_000_000, ConfidenceLevel.HIGH),
    
    # OpenAI models - MEDIUM confidence from industry estimates
    "gpt_4": (1_760_000_000_000, 13_000_000_000_000, ConfidenceLevel.MEDIUM),
    "gpt_4o": (1_760_000_000_000, 13_000_000_000_000, ConfidenceLevel.MEDIUM),  # Similar to GPT-4
    
    # Claude models (Anthropic) - MEDIUM confidence from industry estimates  
    "claude_3.5_sonnet": (250_000_000_000, 10_000_000_000_000, ConfidenceLevel.MEDIUM),
    "claude_3_opus": (175_000_000_000, 8_000_000_000_000, ConfidenceLevel.MEDIUM),
    "claude_opus": (175_000_000_000, 8_000_000_000_000, ConfidenceLevel.MEDIUM),  # Alias
    
    # Gemini models (Google) - MEDIUM confidence from industry estimates
    "gemini_1.5_pro": (300_000_000_000, 12_000_000_000_000, ConfidenceLevel.MEDIUM),
    "gemini_2.0": (500_000_000_000, 20_000_000_000_000, ConfidenceLevel.LOW),  # Speculative
    "gemini_2": (500_000_000_000, 20_000_000_000_000, ConfidenceLevel.LOW),  # Alias
    
    # Qwen models (Alibaba) - MEDIUM confidence from partial disclosures
    "qwen3_235b": (235_000_000_000, 10_000_000_000_000, ConfidenceLevel.MEDIUM),
    "qwen3_480b": (480_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    "qwen3_coder_480b": (480_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    
    # DeepSeek models - MEDIUM confidence from papers
    "deepseek_v3": (671_000_000_000, 14_800_000_000_000, ConfidenceLevel.MEDIUM),  # From paper
    "deepseek": (671_000_000_000, 14_800_000_000_000, ConfidenceLevel.MEDIUM),  # Alias for v3
    "deepseek_r1": (671_000_000_000, 15_000_000_000_000, ConfidenceLevel.LOW),  # Estimated
}

# Benchmark score to FLOP mappings (reference models)
BENCHMARK_REFERENCES = {
    "lmarena_score": {
        1300: 3.8e25,   # Llama 3.1 405B level
        1250: 2.7e25,   # Gemini 1.5 Pro level
        1200: 2.15e25,  # GPT-4 level
        1150: 1.7e25,   # Claude 3.5 Sonnet level
        1100: 1e25,     # Threshold level
        1050: 5e24,     # Sub-threshold
        1000: 1e24,     # Much smaller models
    }
}


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def extract_model_size(model_name: str) -> Optional[int]:
    """Extract parameter count from model name."""
    # Look for patterns like 405B, 70B, 7B, etc.
    patterns = [
        r'(\d+\.?\d*)b(?:illion)?',  # 405B, 7B
        r'(\d+\.?\d*)t(?:rillion)?',  # 1.7T
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_name.lower())
        if match:
            size = float(match.group(1))
            if 't' in pattern:
                return int(size * 1_000_000_000_000)
            else:
                return int(size * 1_000_000_000)
    
    return None


def estimate_training_tokens(model_name: str, param_count: int) -> Tuple[int, ConfidenceLevel, str]:
    """
    Estimate training tokens based on model characteristics and era.
    
    Args:
        model_name: Name of the model 
        param_count: Parameter count in absolute numbers
        
    Returns:
        Tuple of (estimated_tokens, confidence_level, reasoning_suffix)
    """
    model_lower = model_name.lower()
    
    # Modern era models (2023+) - higher token counts, better ratios
    if any(x in model_lower for x in ['llama_3', 'llama_4', 'qwen3', 'gemma_3', 'claude_3', 'gpt_4']):
        # Modern models use ~15-20x tokens per parameter
        if param_count >= 100_000_000_000:  # >= 100B params
            tokens = param_count * 15  # Conservative for large models
            confidence = ConfidenceLevel.MEDIUM
            reasoning = f"(Modern large model: {param_count/1e9:.0f}B params * 15 tokens/param)"
        else:
            tokens = param_count * 20  # Higher ratio for smaller modern models
            confidence = ConfidenceLevel.MEDIUM  
            reasoning = f"(Modern model: {param_count/1e9:.0f}B params * 20 tokens/param)"
    
    # Mid-era models (2022-2023) - moderate token counts
    elif any(x in model_lower for x in ['llama_2', 'qwen2', 'gemma_2', 'mistral', 'claude_2']):
        tokens = param_count * 12  # Moderate training ratios
        confidence = ConfidenceLevel.MEDIUM
        reasoning = f"(Mid-era model: {param_count/1e9:.0f}B params * 12 tokens/param)"
    
    # Earlier era models (2021-2022) - lower token counts
    elif any(x in model_lower for x in ['llama_1', 'llama_13b', 'llama_7b', 'gpt_3', 'palm']):
        tokens = param_count * 8  # Earlier models had lower token ratios
        confidence = ConfidenceLevel.LOW
        reasoning = f"(Early era model: {param_count/1e9:.0f}B params * 8 tokens/param)"
    
    # Specialized models - different training patterns
    elif any(x in model_lower for x in ['coder', 'code', 'instruct', 'chat']):
        tokens = param_count * 18  # Code/instruct models often have higher ratios
        confidence = ConfidenceLevel.MEDIUM
        reasoning = f"(Specialized model: {param_count/1e9:.0f}B params * 18 tokens/param)"
    
    # Chinese models - often have different training ratios
    elif any(x in model_lower for x in ['qwen', 'glm', 'chatglm', 'baichuan', 'yi']):
        tokens = param_count * 16  # Chinese models typically well-trained
        confidence = ConfidenceLevel.MEDIUM
        reasoning = f"(Chinese model: {param_count/1e9:.0f}B params * 16 tokens/param)"
    
    # Default fallback - conservative estimate
    else:
        tokens = param_count * 15  # Conservative middle ground
        confidence = ConfidenceLevel.LOW
        reasoning = f"(Generic estimate: {param_count/1e9:.0f}B params * 15 tokens/param)"
    
    return tokens, confidence, reasoning


def find_manual_override(model_name: str) -> Optional[Tuple[float, ConfidenceLevel, str]]:
    """Find a manual override from Epoch AI's tracker data."""
    model_lower = model_name.lower()
    
    # Remove version suffixes and clean up
    cleaned = re.sub(r'[-_]instruct$|[-_]chat$|[-_]base$', '', model_lower)
    cleaned = re.sub(r'[-_]fp\d+$|[-_]bnb[-_]nf\d+', '', cleaned)
    
    # Find all matching patterns and prefer the longest (most specific) one
    matches = []
    for pattern, (flop, confidence, reasoning) in MANUAL_OVERRIDES.items():
        if pattern in cleaned:
            matches.append((len(pattern), pattern, flop, confidence, reasoning))
    
    if matches:
        # Sort by pattern length (descending) to prefer most specific match
        matches.sort(reverse=True)
        _, pattern, flop, confidence, reasoning = matches[0]
        return (flop, confidence, reasoning)
            
    return None


def find_known_model_match(model_name: str) -> Optional[Tuple[int, int, ConfidenceLevel]]:
    """Find a matching known model specification."""
    model_lower = model_name.lower()
    
    # Remove version suffixes and clean up
    cleaned = re.sub(r'[-_]instruct$|[-_]chat$|[-_]base$', '', model_lower)
    cleaned = re.sub(r'[-_]fp\d+$|[-_]bnb[-_]nf\d+', '', cleaned)
    
    for pattern, specs in KNOWN_MODEL_SPECS.items():
        if pattern in cleaned:
            return specs
            
    return None


def estimate_from_benchmark_score(model: Model, estimator: ComputeEstimator) -> Optional[Dict]:
    """Estimate FLOP from benchmark scores."""
    # Check for available benchmark scores (prioritize OpenLM Arena ELO)
    elo_score = model.benchmarks.get('openlm_arena_elo') or model.benchmarks.get('lmarena_score')
    if not elo_score:
        return None
        
    # Find closest reference point
    references = BENCHMARK_REFERENCES['lmarena_score']
    sorted_refs = sorted(references.items(), key=lambda x: abs(x[0] - elo_score))
    
    if not sorted_refs:
        return None
        
    ref_score, ref_flop = sorted_refs[0]
    score_diff = abs(ref_score - elo_score)
    
    # Use ELO-based estimation
    try:
        result = estimator.estimate_from_benchmark_elo(
            elo_rating=elo_score,
            reference_elo=ref_score,
            reference_flop=ref_flop
        )
        
        # Adjust confidence based on score difference
        if score_diff > 100:
            result.confidence = ConfidenceLevel.LOW
        elif score_diff > 50:
            result.confidence = ConfidenceLevel.MEDIUM
            
        benchmark_type = 'openlm_arena_elo' if 'openlm_arena_elo' in model.benchmarks else 'lmarena_score'
        
        return {
            'flop': result.flop_estimate,
            'confidence': result.confidence,
            'method': result.method,
            'reasoning': f"Benchmark-based ({benchmark_type}): {result.reasoning}"
        }
    except Exception as e:
        logging.warning(f"Failed to estimate from benchmark for {model.name}: {e}")
        return None


def deduplicate_models(models: List[Model]) -> List[Model]:
    """
    Deduplicate models by name, keeping the one with highest training FLOP.
    
    For models with same name, preference order:
    1. Highest training_flop value
    2. If same FLOP, highest confidence level
    3. If same confidence, most recent last_updated
    
    Args:
        models: List of models potentially containing duplicates
        
    Returns:
        List of deduplicated models
    """
    seen = {}
    confidence_order = {
        ConfidenceLevel.HIGH: 4,
        ConfidenceLevel.MEDIUM: 3, 
        ConfidenceLevel.LOW: 2,
        ConfidenceLevel.SPECULATIVE: 1
    }
    
    for model in models:
        key = model.name.lower().strip()
        
        if key not in seen:
            seen[key] = model
        else:
            existing = seen[key]
            
            # Compare training FLOP (prefer higher)
            model_flop = model.training_flop or 0
            existing_flop = existing.training_flop or 0
            
            if model_flop > existing_flop:
                seen[key] = model
            elif model_flop == existing_flop:
                # Same FLOP, compare confidence
                model_conf = confidence_order.get(model.training_flop_confidence, 0)
                existing_conf = confidence_order.get(existing.training_flop_confidence, 0)
                
                if model_conf > existing_conf:
                    seen[key] = model
                elif model_conf == existing_conf:
                    # Same confidence, prefer more recent
                    if model.last_updated > existing.last_updated:
                        seen[key] = model
    
    deduplicated = list(seen.values())
    logging.info(f"Deduplicated {len(models)} models to {len(deduplicated)} unique models")
    
    return deduplicated


def estimate_model_flops(model: Model, estimator: ComputeEstimator) -> Optional[Dict]:
    """
    Estimate FLOP for a single model using hierarchical methods with multiple estimates.
    
    Now collects ALL applicable estimation methods and stores alternatives.
    
    Method Priority (Best to Worst):
    0. Manual Overrides - Curated values from Epoch AI tracker (HIGH/MEDIUM confidence)
    1. Known Model Specifications - Use published/official specifications (HIGH confidence)
    2. Parameter-based Scaling Laws - Extract param count from name, use Chinchilla (MEDIUM confidence)
    3. Benchmark Score Interpolation - ELO-based estimates (MEDIUM/LOW confidence) 
    
    Args:
        model: Model object with name and benchmark data
        estimator: ComputeEstimator instance
        
    Returns:
        Dict with primary estimate and alternative estimates, or None if no estimate possible
    """
    all_estimates = []
    primary_estimate = None
    
    # PRIORITY 0: Manual overrides (HIGHEST PRIORITY - Epoch AI tracker data)
    manual_override = find_manual_override(model.name)
    if manual_override:
        flop, confidence, reasoning = manual_override
        estimate = {
            'flop': flop,
            'confidence': confidence,
            'method': EstimationMethod.EPOCH_ESTIMATE,
            'reasoning': reasoning,
            'priority': 0
        }
        all_estimates.append(estimate)
        if primary_estimate is None:
            primary_estimate = estimate
    
    # PRIORITY 1: Known model specifications
    known_specs = find_known_model_match(model.name)
    if known_specs:
        params, tokens, confidence = known_specs
        
        # Update model parameters if not already set
        if model.parameters is None:
            model.parameters = params
            model.parameter_source = "known_specification"
        
        result = estimator.estimate_from_scaling_laws(params, tokens, "chinchilla")
        estimate = {
            'flop': result.flop_estimate,
            'confidence': confidence,
            'method': result.method,
            'reasoning': f"Scaling laws with documented parameters: {result.reasoning}",
            'parameters': params,
            'parameter_source': 'known_specification',
            'priority': 1
        }
        all_estimates.append(estimate)
        if primary_estimate is None:
            primary_estimate = estimate
    
    # PRIORITY 2: Parameter-based Chinchilla scaling
    param_count = extract_model_size(model.name)
    if param_count and not any(est.get('parameters') == param_count for est in all_estimates):
        # Only add if we don't already have this parameter count
        if model.parameters is None:
            model.parameters = param_count
            model.parameter_source = "extracted_from_name"
        
        estimated_tokens, confidence, reasoning_suffix = estimate_training_tokens(model.name, param_count)
        result = estimator.estimate_from_scaling_laws(param_count, estimated_tokens, "chinchilla")
        estimate = {
            'flop': result.flop_estimate,
            'confidence': confidence,
            'method': result.method,
            'reasoning': f"Parameter-based Chinchilla scaling: {result.reasoning} {reasoning_suffix}",
            'parameters': param_count,
            'parameter_source': 'extracted_from_name',
            'priority': 2
        }
        all_estimates.append(estimate)
        if primary_estimate is None:
            primary_estimate = estimate
    
    # PRIORITY 3: Benchmark-based estimation
    benchmark_estimate = estimate_from_benchmark_score(model, estimator)
    if benchmark_estimate:
        estimate = {
            'flop': benchmark_estimate['flop'],
            'confidence': benchmark_estimate['confidence'],
            'method': benchmark_estimate['method'],
            'reasoning': benchmark_estimate['reasoning'],
            'priority': 3
        }
        all_estimates.append(estimate)
        if primary_estimate is None:
            primary_estimate = estimate
    
    if not all_estimates:
        return None
    
    # Sort by priority and add alternative estimates to the model
    all_estimates.sort(key=lambda x: x['priority'])
    
    # Add alternative estimates (excluding the primary one)
    model.alternative_estimates.clear()  # Clear existing alternatives
    for est in all_estimates[1:]:  # Skip the primary estimate
        model.add_alternative_estimate(
            flop=est['flop'],
            confidence=est['confidence'],
            method=est['method'],
            reasoning=est['reasoning']
        )
    
    # Return the primary estimate
    result = primary_estimate.copy()
    result.pop('priority', None)  # Remove priority from result
    return result


def main():
    """Main entry point for FLOP estimation."""
    parser = argparse.ArgumentParser(
        description="Apply FLOP estimations to scraped models"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update the original data files with FLOP estimates"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without updating files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting FLOP estimation for scraped models...")
    
    # Initialize components
    storage = JSONStorage(args.data_dir)
    estimator = ComputeEstimator()
    
    # Load all models from scraped data
    raw_models = storage.load_all_scraped_models()
    logger.info(f"Loaded {len(raw_models)} raw models from scraped data")
    
    # Deduplicate models BEFORE processing to avoid counting duplicates
    logger.info("Deduplicating models before processing...")
    all_models = deduplicate_models(raw_models)
    logger.info(f"Deduplicated to {len(all_models)} unique models for processing")
    
    # Track statistics
    updated_count = 0
    above_threshold_count = 0
    
    # Process each unique model
    for model in all_models:
        if model.training_flop and not args.update:
            # Skip if already has FLOP estimate and not forcing update
            continue
            
        estimate = estimate_model_flops(model, estimator)
        if estimate:
            if args.dry_run:
                logger.info(f"Would update {model.name}: {estimate['flop']:.2e} FLOP "
                          f"(confidence: {estimate['confidence'].value})")
            else:
                # Update the model with threshold classification
                model.update_flop_estimate(
                    flop=estimate['flop'],
                    confidence=estimate['confidence'],
                    method=estimate['method'],
                    reasoning=estimate['reasoning'],
                    high_confidence_above_threshold=HIGH_CONFIDENCE_ABOVE_THRESHOLD,
                    high_confidence_below_threshold=HIGH_CONFIDENCE_BELOW_THRESHOLD
                )
                
                # Update parameters if provided in estimate
                if 'parameters' in estimate and model.parameters is None:
                    model.parameters = estimate['parameters']
                    model.parameter_source = estimate.get('parameter_source', 'extracted_from_name')
                
                updated_count += 1
                
                if estimate['flop'] >= 1e25:
                    above_threshold_count += 1
                    logger.info(f"Updated {model.name}: {estimate['flop']:.2e} FLOP "
                              f"(ABOVE THRESHOLD, confidence: {estimate['confidence'].value})")
                else:
                    logger.debug(f"Updated {model.name}: {estimate['flop']:.2e} FLOP "
                               f"(confidence: {estimate['confidence'].value})")
    
    # Apply threshold classification to all models (including those not updated)
    if not args.dry_run:
        logger.info("Applying threshold classification to all models...")
        classification_count = 0
        
        for model in all_models:
            if model.training_flop is not None:
                old_classification = model.threshold_classification
                new_classification = model.classify_by_threshold(
                    HIGH_CONFIDENCE_ABOVE_THRESHOLD, 
                    HIGH_CONFIDENCE_BELOW_THRESHOLD
                )
                
                if old_classification != new_classification:
                    model.threshold_classification = new_classification
                    classification_count += 1
                    logger.debug(f"Classified {model.name}: {new_classification.value}")
        
        logger.info(f"Applied threshold classification to {classification_count} models")
    
    if not args.dry_run:
        # Save models to consolidated file (already deduplicated)
        logger.info("Saving models to consolidated estimated file...")
        
        # Create consolidated collection
        collection = ModelCollection(
            models=all_models, 
            source="consolidated_estimated"
        )
        
        # Save to estimated directory
        filepath = storage.save_models(collection, "estimated_models", stage="estimated")
        logger.info(f"Saved {len(all_models)} models to {filepath}")
    
    # Calculate threshold classification statistics using deduplicated models
    final_models = all_models
    models_with_flop = [m for m in final_models if m.training_flop is not None]
    high_confidence_above = len([m for m in models_with_flop if m.threshold_classification == ThresholdClassification.HIGH_CONFIDENCE_ABOVE])
    high_confidence_below = len([m for m in models_with_flop if m.threshold_classification == ThresholdClassification.HIGH_CONFIDENCE_BELOW])
    not_sure = len([m for m in models_with_flop if m.threshold_classification == ThresholdClassification.NOT_SURE])
    
    # Print summary
    print(f"\n{'='*60}")
    print("FLOP ESTIMATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total models processed: {len(all_models)}")
    if not args.dry_run:
        print(f"Unique models after deduplication: {len(all_models)}")
    print(f"Models with FLOP estimates: {len(models_with_flop)}")
    print(f"Models updated this run: {updated_count}")
    print(f"Models above 1e25 FLOP: {above_threshold_count}")
    
    print(f"\nTHRESHOLD CLASSIFICATION:")
    print(f"  High confidence > 1e25 FLOP (>= {HIGH_CONFIDENCE_ABOVE_THRESHOLD:.1e}): {high_confidence_above}")
    print(f"  High confidence < 1e25 FLOP (<= {HIGH_CONFIDENCE_BELOW_THRESHOLD:.1e}): {high_confidence_below}")
    print(f"  Not sure ({HIGH_CONFIDENCE_BELOW_THRESHOLD:.1e} - {HIGH_CONFIDENCE_ABOVE_THRESHOLD:.1e}): {not_sure}")
    
    if high_confidence_above > 0:
        print(f"\nHigh confidence models > 1e25 FLOP:")
        high_conf_above_models = [m for m in models_with_flop if m.threshold_classification == ThresholdClassification.HIGH_CONFIDENCE_ABOVE]
        for model in sorted(high_conf_above_models, key=lambda m: m.training_flop, reverse=True)[:10]:
            print(f"  - {model.name}: {model.training_flop:.2e} FLOP "
                  f"({model.training_flop_confidence.value})")
    
    if above_threshold_count > 0 and above_threshold_count != high_confidence_above:
        print(f"\nAll models likely above 1e25 FLOP threshold:")
        above_threshold = [m for m in all_models if m.training_flop and m.training_flop >= 1e25]
        for model in sorted(above_threshold, key=lambda m: m.training_flop, reverse=True)[:10]:
            print(f"  - {model.name}: {model.training_flop:.2e} FLOP "
                  f"({model.training_flop_confidence.value}) [{model.threshold_classification.value}]")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    sys.exit(main())