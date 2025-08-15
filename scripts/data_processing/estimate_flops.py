#!/usr/bin/env python3
"""
Apply FLOP estimations to scraped models.

This script processes models from the data store and estimates their training FLOP
using various methods including benchmark-based estimation, known model mappings,
and heuristics based on model names and metadata.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from epoch_tracker.estimation import ComputeEstimator
from epoch_tracker.models import Model, ModelCollection, ConfidenceLevel, EstimationMethod, ModelStatus
from epoch_tracker.storage import JSONStorage
from epoch_tracker.utils.developer_exclusion import DeveloperExclusion
from epoch_tracker.config.thresholds import get_threshold_config


# Global threshold configuration - loaded once at startup
THRESHOLD_CONFIG = None

# Manual overrides based on Epoch AI's tracker (https://epoch.ai/data-insights/models-over-1e25-flop)
# These take HIGHEST PRIORITY and override all other estimation methods
# NOTE: Use exact FLOP values from Epoch's research with appropriate confidence levels
MANUAL_OVERRIDES = {
    # All manual overrides use HIGH confidence since they represent our most authoritative estimates
    # But descriptions preserve the original precision assessment from Epoch research
    
    # High-precision estimates from Epoch AI
    "llama_3.1_405b": (3.8e25, ConfidenceLevel.HIGH, "Manual override: High-precision estimate from Meta disclosure"),
    "grok_2": (3.0e25, ConfidenceLevel.HIGH, "Manual override: High-precision estimate from xAI disclosure"),
    
    # Low-precision but reliable estimates from Epoch AI
    "claude_3_opus": (1.6e25, ConfidenceLevel.HIGH, "Manual override: Low-precision estimate from industry analysis"),
    "claude_opus": (1.6e25, ConfidenceLevel.HIGH, "Manual override: Low-precision estimate from industry analysis"),  # Alias
    "gpt_4": (2.1e25, ConfidenceLevel.HIGH, "Manual override: Low-precision estimate from scaling analysis"),
    "gemini_1.0_ultra": (5.0e25, ConfidenceLevel.HIGH, "Manual override: Low-precision estimate from Google hints"),
    "gemini_ultra": (5.0e25, ConfidenceLevel.HIGH, "Manual override: Low-precision estimate from Google hints"),  # Alias
    "claude_3.5_sonnet": (3.6e25, ConfidenceLevel.HIGH, "Manual override: Low-precision estimate from benchmarks"),
    "gpt_4o": (3.8e25, ConfidenceLevel.HIGH, "Manual override: Low-precision estimate from OpenAI patterns"),
    
    # Speculative estimates from Epoch AI
    "claude_opus_4": (1.5e26, ConfidenceLevel.HIGH, "Manual override: Speculative estimate for next-gen Claude"),
    "claude_4_opus": (1.5e26, ConfidenceLevel.HIGH, "Manual override: Speculative estimate for next-gen Claude"),  # Alias
    "gpt_4.5": (6.4e25, ConfidenceLevel.HIGH, "Manual override: Speculative estimate for GPT-4.5"),
    "gpt_4.5_preview": (6.4e25, ConfidenceLevel.HIGH, "Manual override: Speculative estimate for GPT-4.5"),  # Alias

    # High-precision estimates from other sources
    "deepseek_r1": (3.0e24, ConfidenceLevel.HIGH, "Manual override: High-precision estimate from DeepSeek disclosure"),
}

# Known model specifications for direct FLOP calculation
# NOTE: Patterns must match normalized model names (underscores, not hyphens)
# To add new models: Include normalized name pattern, parameters, training tokens, confidence level
KNOWN_MODEL_SPECS = {
    # All Chinchilla scaling estimates use MEDIUM confidence regardless of disclosure quality
    # This standardizes confidence across all parameter-based scaling methods
    
    # Llama models (Meta) - From official disclosures
    "llama_3.1_405b": (405_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    "llama_3.1_70b": (70_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    "llama_3.1_8b": (8_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    "llama_3_405b": (405_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    "llama_3_70b": (70_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    "llama_3_8b": (8_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    "llama_2_70b": (70_000_000_000, 2_000_000_000_000, ConfidenceLevel.MEDIUM),
    "llama_2_7b": (7_000_000_000, 2_000_000_000_000, ConfidenceLevel.MEDIUM),
    
    # OpenAI models - From industry estimates
    "gpt_4": (1_760_000_000_000, 13_000_000_000_000, ConfidenceLevel.MEDIUM),
    "gpt_4o": (1_760_000_000_000, 13_000_000_000_000, ConfidenceLevel.MEDIUM),  # Similar to GPT-4
    
    # Claude models (Anthropic) - From industry estimates  
    "claude_3.5_sonnet": (250_000_000_000, 10_000_000_000_000, ConfidenceLevel.MEDIUM),
    "claude_3_opus": (175_000_000_000, 8_000_000_000_000, ConfidenceLevel.MEDIUM),
    "claude_opus": (175_000_000_000, 8_000_000_000_000, ConfidenceLevel.MEDIUM),  # Alias
    
    # Gemini models (Google) - From industry estimates
    "gemini_1.5_pro": (300_000_000_000, 12_000_000_000_000, ConfidenceLevel.MEDIUM),
    # Note: Gemini 2.0+ models removed - no official parameter disclosure available
    
    # Qwen models (Alibaba) - From partial disclosures
    "qwen3_235b": (235_000_000_000, 10_000_000_000_000, ConfidenceLevel.MEDIUM),
    "qwen3_480b": (480_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    "qwen3_coder_480b": (480_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),
    
    # DeepSeek models - From papers and disclosures  
    "deepseek_v3": (671_000_000_000, 14_800_000_000_000, ConfidenceLevel.MEDIUM),  # From paper
    "deepseek_r1": (671_000_000_000, 15_000_000_000_000, ConfidenceLevel.MEDIUM),  # All specs use MEDIUM
}

def load_estimation_methods_config(config_path: Optional[Path] = None) -> Dict:
    """Load FLOP estimation methods configuration from JSON file.
    
    Args:
        config_path: Path to flop estimation methods JSON file
        
    Returns:
        Dictionary of estimation methods configuration
    """
    if config_path is None:
        # Default path relative to script location
        script_dir = Path(__file__).parent
        config_path = script_dir.parent.parent / "configs" / "flop_estimation_methods.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded FLOP estimation methods configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"FLOP estimation methods config not found at {config_path}")
        raise
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error loading FLOP estimation methods config: {e}")
        raise


def load_benchmark_references(config_path: Optional[Path] = None) -> Dict:
    """Load benchmark reference models from JSON configuration file.
    
    Args:
        config_path: Path to benchmark references JSON file
        
    Returns:
        Dictionary of benchmark reference configurations
    """
    if config_path is None:
        # Default path relative to script location
        script_dir = Path(__file__).parent
        config_path = script_dir.parent.parent / "configs" / "benchmark_references.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config["benchmark_references"]
    except FileNotFoundError:
        logging.warning(f"Benchmark references config not found at {config_path}, using fallback")
        return get_fallback_benchmark_references()
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error loading benchmark references config: {e}, using fallback")
        return get_fallback_benchmark_references()


def infer_developer_from_references(model: Model) -> str:
    """Infer the correct developer from benchmark reference data."""
    if not model.name:
        return model.developer
        
    # Check if this model is a reference model in any benchmark
    for benchmark_name, ref_config in BENCHMARK_REFERENCE_MODELS.items():
        if model.name.lower() == ref_config["reference_model"].lower():
            source = ref_config.get("source", "")
            # Extract known company names from source descriptions
            if "openai" in source.lower():
                return "OpenAI"
            elif "anthropic" in source.lower():
                return "Anthropic"
            elif "google" in source.lower():
                return "Google"
            elif "meta" in source.lower():
                return "Meta"
            elif "deepseek" in source.lower():
                return "DeepSeek"
    
    # Keep original developer if no inference possible
    return model.developer


def get_fallback_benchmark_references() -> Dict:
    """Fallback benchmark references if JSON config fails to load."""
    return {
        "lmarena_score": {
            "reference_model": "llama_3.1_405b",
            "reference_score": 1335.0,
            "reference_flop": 3.8e25,
            "source": "LMArena Manual Data Collection + Epoch AI disclosure analysis",
            "scaling_exponent": 3.0
        },
        "openlm_arena_elo": {
            "reference_model": "llama_3.1_405b", 
            "reference_score": 1286.0,
            "reference_flop": 3.8e25,
            "source": "OpenLM Arena leaderboard + Epoch AI disclosure analysis",
            "scaling_exponent": 3.0
        }
    }


# Load benchmark reference models from JSON configuration
# To add new benchmarks, edit configs/benchmark_references.json
BENCHMARK_REFERENCE_MODELS = load_benchmark_references()

# Configuration loading - will be set after all functions are defined
ESTIMATION_METHODS_CONFIG = None


def get_manual_overrides() -> Dict:
    """Get manual overrides from configuration."""
    if ESTIMATION_METHODS_CONFIG is None:
        return {}
    return ESTIMATION_METHODS_CONFIG.get("manual_overrides", {}).get("entries", {})


def get_known_model_specs() -> Dict:
    """Get known model specifications from configuration."""
    if ESTIMATION_METHODS_CONFIG is None:
        return {}
    return ESTIMATION_METHODS_CONFIG.get("known_model_specifications", {}).get("entries", {})


def get_estimation_methods() -> List[Dict]:
    """Get estimation methods in priority order."""
    if ESTIMATION_METHODS_CONFIG is None:
        return []
    return ESTIMATION_METHODS_CONFIG.get("method_priority_order", [])


# get_threshold_config function removed - now using centralized version from epoch_tracker.config.thresholds


class MethodUsageTracker:
    """Track usage statistics for each estimation method."""
    
    def __init__(self):
        self.stats = {}
        self.total_models = 0
        self.models_with_estimates = 0
        
    def record_method_usage(self, method_id: str, model_name: str, confidence: ConfidenceLevel, flop: float):
        """Record usage of an estimation method."""
        if method_id not in self.stats:
            self.stats[method_id] = {
                'count': 0,
                'models': [],
                'confidence_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'SPECULATIVE': 0},
                'above_threshold': 0,
                'total_flop': 0.0
            }
        
        self.stats[method_id]['count'] += 1
        self.stats[method_id]['models'].append({
            'name': model_name,
            'confidence': confidence,
            'flop': flop
        })
        self.stats[method_id]['confidence_distribution'][confidence.name] += 1
        self.stats[method_id]['total_flop'] += flop
        
        if flop >= 1e25:
            self.stats[method_id]['above_threshold'] += 1
    
    def record_no_estimate(self, model_name: str):
        """Record that no estimate was possible for a model."""
        if 'no_estimate' not in self.stats:
            self.stats['no_estimate'] = {
                'count': 0,
                'models': []
            }
        self.stats['no_estimate']['count'] += 1
        self.stats['no_estimate']['models'].append(model_name)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_models': self.total_models,
            'models_with_estimates': self.models_with_estimates,
            'method_stats': self.stats
        }

# Legacy benchmark references (kept for backwards compatibility)
BENCHMARK_REFERENCES = {
    "lmarena_score": {
        1300: 3.8e25,   # Legacy reference point
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


def initialize_configuration():
    """Initialize the estimation methods configuration and constants."""
    global ESTIMATION_METHODS_CONFIG, THRESHOLD_CONFIG
    
    # Load the configuration
    ESTIMATION_METHODS_CONFIG = load_estimation_methods_config()
    
    # Load centralized threshold configuration
    THRESHOLD_CONFIG = get_threshold_config()


def print_enhanced_summary(all_models: List[Model], updated_count: int, above_threshold_count: int, 
                          usage_tracker: MethodUsageTracker, args):
    """Print enhanced FLOP estimation summary with method priority and usage statistics."""
    final_models = all_models
    models_with_flop = [m for m in final_models if m.training_flop is not None]
    confirmed_above = len([m for m in models_with_flop if m.status == ModelStatus.CONFIRMED_ABOVE])
    likely_above = len([m for m in models_with_flop if m.status == ModelStatus.LIKELY_ABOVE])
    confirmed_below = len([m for m in models_with_flop if m.status == ModelStatus.CONFIRMED_BELOW])
    likely_below = len([m for m in models_with_flop if m.status == ModelStatus.LIKELY_BELOW])
    uncertain = len([m for m in models_with_flop if m.status == ModelStatus.UNCERTAIN])
    
    print(f"\n{'='*80}")
    print("FLOP ESTIMATION SUMMARY")
    print(f"{'='*80}")
    
    # Method Priority Order Section
    print("\nEstimation Methods (Priority Order):")
    estimation_methods = get_estimation_methods()
    for i, method in enumerate(estimation_methods, 1):
        enabled_status = "✓" if method.get("enabled", True) else "✗"
        confidence_levels = "/".join(method.get("confidence_levels", []))
        print(f"  {i}. {method['name']} [{enabled_status}] - {method['description']}")
        print(f"     Confidence: {confidence_levels} | Source: {method['source_description']}")
    
    print(f"\nMethod Usage Statistics:")
    usage_stats = usage_tracker.get_summary()
    
    # Show method-specific statistics
    method_name_map = {
        "manual_overrides": "Manual Overrides",
        "known_specifications": "Known Model Specifications", 
        "parameter_based_scaling": "Parameter-based Chinchilla Scaling",
        "benchmark_interpolation": "Benchmark Score Interpolation"
    }
    
    total_estimated = 0
    for method_id, display_name in method_name_map.items():
        if method_id in usage_stats['method_stats']:
            stats = usage_stats['method_stats'][method_id]
            count = stats['count']
            above_threshold = stats['above_threshold']
            total_estimated += count
            
            confidence_dist = stats['confidence_distribution']
            conf_summary = f"H:{confidence_dist['HIGH']} M:{confidence_dist['MEDIUM']} L:{confidence_dist['LOW']} S:{confidence_dist['SPECULATIVE']}"
            
            print(f"  • {display_name}: {count} models ({above_threshold} above 1e25 FLOP)")
            print(f"    Confidence distribution: {conf_summary}")
            
            # Show top models for this method
            if count > 0:
                top_models = sorted(stats['models'], key=lambda x: x['flop'], reverse=True)[:3]
                model_names = [f"{m['name']} ({m['flop']:.1e})" for m in top_models]
                print(f"    Top models: {', '.join(model_names)}")
        else:
            print(f"  • {display_name}: 0 models")
    
    # Show no estimate statistics
    if 'no_estimate' in usage_stats['method_stats']:
        no_est_count = usage_stats['method_stats']['no_estimate']['count']
        print(f"  • No estimate possible: {no_est_count} models")
    
    print(f"\nOverall Statistics:")
    print(f"  Total models processed: {len(all_models)}")
    if not args.dry_run:
        print(f"  Unique models after deduplication: {len(all_models)}")
    print(f"  Models with FLOP estimates: {len(models_with_flop)}")
    print(f"  Models updated this run: {updated_count}")
    print(f"  Models above 1e25 FLOP: {above_threshold_count}")
    
    print(f"\nModel Status Classification:")
    print(f"  Confirmed above 1e25 FLOP: {confirmed_above}")
    print(f"  Likely above 1e25 FLOP: {likely_above}")
    print(f"  Uncertain: {uncertain}")
    print(f"  Likely below 1e25 FLOP: {likely_below}")
    print(f"  Confirmed below 1e25 FLOP: {confirmed_below}")
    
    # Show example models above threshold
    if confirmed_above + likely_above > 0:
        print(f"\nModels above 1e25 FLOP:")
        above_models = [m for m in models_with_flop if m.status in [ModelStatus.CONFIRMED_ABOVE, ModelStatus.LIKELY_ABOVE]]
        for model in sorted(above_models, key=lambda m: m.training_flop, reverse=True)[:10]:
            print(f"  - {model.name}: {model.training_flop:.2e} FLOP "
                  f"({model.training_flop_confidence.value}, {model.status.value})")
    
    if above_threshold_count > 0 and above_threshold_count != confirmed_above + likely_above:
        print(f"\nAll models likely above 1e25 FLOP threshold:")
        above_threshold = [m for m in all_models if m.training_flop and m.training_flop >= 1e25]
        for model in sorted(above_threshold, key=lambda m: m.training_flop, reverse=True)[:10]:
            print(f"  - {model.name}: {model.training_flop:.2e} FLOP "
                  f"({model.training_flop_confidence.value}) [{model.status.value}]")
    
    print(f"{'='*80}\n")


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
    All parameter-based scaling uses MEDIUM confidence consistently.
    
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
        confidence = ConfidenceLevel.MEDIUM  # All parameter-based scaling uses MEDIUM
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
        confidence = ConfidenceLevel.MEDIUM  # All parameter-based scaling uses MEDIUM
        reasoning = f"(Generic estimate: {param_count/1e9:.0f}B params * 15 tokens/param)"
    
    return tokens, confidence, reasoning


def find_manual_override(model_name: str) -> Optional[Tuple[float, ConfidenceLevel, str]]:
    """Find a manual override from Epoch AI's tracker data."""
    model_lower = model_name.lower()
    
    # Remove version suffixes and clean up
    cleaned = re.sub(r'[-_]instruct$|[-_]chat$|[-_]base$', '', model_lower)
    cleaned = re.sub(r'[-_]fp\d+$|[-_]bnb[-_]nf\d+', '', cleaned)
    
    # Get manual overrides from configuration
    manual_overrides = get_manual_overrides()
    
    # Find exact matching patterns only (no more fuzzy matching)
    matches = []
    for pattern, override_data in manual_overrides.items():
        if pattern == cleaned:
            flop = override_data.get("training_flop")
            confidence_str = override_data.get("confidence", "MEDIUM")
            reasoning = override_data.get("reasoning", "Manual override")
            
            # Convert confidence string to enum
            confidence = getattr(ConfidenceLevel, confidence_str, ConfidenceLevel.MEDIUM)
            
            matches.append((len(pattern), pattern, flop, confidence, reasoning))
    
    if matches:
        # Sort by pattern length (descending) to prefer most specific match
        matches.sort(reverse=True)
        _, pattern, flop, confidence, reasoning = matches[0]
        return (flop, confidence, reasoning)
            
    return None


def find_known_model_match(model_name: str) -> Optional[Tuple[int, int, ConfidenceLevel, str]]:
    """Find a matching known model specification.
    
    Returns:
        Tuple of (parameters, training_tokens, confidence_level, matched_pattern) or None
    """
    model_lower = model_name.lower()
    
    # Remove version suffixes and clean up
    cleaned = re.sub(r'[-_]instruct$|[-_]chat$|[-_]base$', '', model_lower)
    cleaned = re.sub(r'[-_]fp\d+$|[-_]bnb[-_]nf\d+', '', cleaned)
    
    # Get known model specifications from configuration
    known_specs = get_known_model_specs()
    
    for pattern, spec_data in known_specs.items():
        if pattern == cleaned:
            parameters = spec_data.get("parameters")
            training_tokens = spec_data.get("training_tokens")
            confidence_str = spec_data.get("confidence", "MEDIUM")
            
            # Convert confidence string to enum
            confidence = getattr(ConfidenceLevel, confidence_str, ConfidenceLevel.MEDIUM)
            
            return (parameters, training_tokens, confidence, pattern)
            
    return None


def estimate_from_single_benchmark(benchmark_name: str, benchmark_score: float, 
                                  estimator: ComputeEstimator, model_name: str = None) -> Optional[Dict]:
    """Estimate FLOP from a single benchmark score using new reference system."""
    if benchmark_name not in BENCHMARK_REFERENCE_MODELS:
        return None
        
    ref_config = BENCHMARK_REFERENCE_MODELS[benchmark_name]
    
    try:
        # Check if this benchmark uses threshold logic (e.g., Sora-based video benchmarks)
        use_threshold_logic = ref_config.get("threshold_logic", False)
        
        if use_threshold_logic:
            # Special threshold logic: models outperforming reference assumed >1e25 FLOP
            reference_score = ref_config["reference_score"]
            reference_flop = ref_config["reference_flop"]  # This should be 1e25 for Sora
            reference_model = ref_config["reference_model"]
            
            # Check if this IS the reference model
            if model_name and model_name.lower() == reference_model.lower():
                # Reference model gets exactly its reference FLOP value
                final_flop = reference_flop  # 1e25 for Sora
                confidence = ConfidenceLevel.MEDIUM
                reasoning = f"Threshold reference model ({benchmark_name}): {reference_model} set to {final_flop:.2e} FLOP as threshold marker"
            elif benchmark_score > reference_score:
                # Model outperforms threshold model (Sora) -> assume >1e25 FLOP
                # Give it a slightly higher estimate to ensure it's above threshold
                final_flop = max(1.1e25, reference_flop * 1.1)
                confidence = ConfidenceLevel.MEDIUM
                reasoning = f"Threshold-based ({benchmark_name}): {benchmark_score} > {reference_model} ({reference_score}) → assumed >1e25 FLOP (reference model assumed frontier-level)"
            else:
                # Model underperforms threshold model -> scale normally but below threshold
                alpha = ref_config.get("scaling_exponent", 2.5)
                flop_ratio = (benchmark_score / reference_score) ** alpha
                scaled_flop = reference_flop * flop_ratio
                
                # Ensure it stays below threshold if scaling suggests it
                # Cap at 99% of the high_confidence_below_threshold to stay clearly below
                cap_value = THRESHOLD_CONFIG.high_confidence_below_threshold * 0.99
                final_flop = min(scaled_flop, cap_value)
                confidence = ConfidenceLevel.LOW
                reasoning = f"Threshold-based ({benchmark_name}): {benchmark_score} < {reference_model} ({reference_score}) → scaled estimate {final_flop:.2e} FLOP (α={alpha})"
            
            return {
                'flop': final_flop,
                'confidence': confidence,
                'method': EstimationMethod.BENCHMARK_BASED,
                'benchmark_name': benchmark_name,
                'reference_model': ref_config["reference_model"],
                'source': ref_config["source"],
                'reasoning': reasoning
            }
        
        else:
            # Normal scaling-based estimation
            result = estimator.estimate_from_benchmark_elo(
                elo_rating=benchmark_score,
                reference_elo=ref_config["reference_score"], 
                reference_flop=ref_config["reference_flop"]
            )
            
            # Override scaling exponent if specified
            if "scaling_exponent" in ref_config:
                alpha = ref_config["scaling_exponent"]
                flop_ratio = (benchmark_score / ref_config["reference_score"]) ** alpha
                result.flop_estimate = ref_config["reference_flop"] * flop_ratio
            
            # Adjust confidence based on distance from reference
            score_diff = abs(benchmark_score - ref_config["reference_score"])
            # All benchmark interpolation uses LOW confidence due to inherent unreliability
            result.confidence = ConfidenceLevel.LOW
            
            return {
                'flop': result.flop_estimate,
                'confidence': result.confidence,
                'method': result.method,
                'benchmark_name': benchmark_name,
                'reference_model': ref_config["reference_model"],
                'source': ref_config["source"],
                'reasoning': f"Benchmark-based ({benchmark_name}): {benchmark_score} vs reference {ref_config['reference_model']} ({ref_config['reference_score']}, {ref_config['reference_flop']:.2e} FLOP) with α={ref_config.get('scaling_exponent', 3.0)} = {result.flop_estimate:.2e} FLOP"
            }
            
    except Exception as e:
        logging.warning(f"Failed to estimate from {benchmark_name} for score {benchmark_score}: {e}")
        return None


def estimate_from_multiple_benchmarks(model: Model, estimator: ComputeEstimator) -> Optional[Dict]:
    """Estimate FLOP from multiple benchmark scores with confidence weighting."""
    if not model.benchmarks:
        return None
    
    # Get estimates from all available benchmarks
    individual_estimates = []
    
    for benchmark_name, score in model.benchmarks.items():
        if benchmark_name in BENCHMARK_REFERENCE_MODELS and score is not None:
            estimate = estimate_from_single_benchmark(benchmark_name, score, estimator, model.name)
            if estimate:
                individual_estimates.append(estimate)
    
    if not individual_estimates:
        return None
    
    # If only one estimate, return it directly
    if len(individual_estimates) == 1:
        return individual_estimates[0]
    
    # Multiple estimates - aggregate with confidence weighting
    confidence_weights = {
        ConfidenceLevel.HIGH: 4.0,
        ConfidenceLevel.MEDIUM: 3.0, 
        ConfidenceLevel.LOW: 2.0,
        ConfidenceLevel.SPECULATIVE: 1.0
    }
    
    total_weight = 0
    weighted_flop_sum = 0
    best_confidence = ConfidenceLevel.SPECULATIVE
    benchmark_details = []
    
    for estimate in individual_estimates:
        weight = confidence_weights[estimate['confidence']]
        weighted_flop_sum += estimate['flop'] * weight
        total_weight += weight
        
        # Track best confidence level
        if estimate['confidence'].value < best_confidence.value:
            best_confidence = estimate['confidence']
            
        benchmark_details.append(f"{estimate['benchmark_name']}: {estimate['flop']:.2e} ({estimate['confidence'].value.title()})")
    
    # Calculate weighted average
    avg_flop = weighted_flop_sum / total_weight
    
    # Boost confidence if multiple benchmarks agree (within 2x of each other)
    flop_values = [est['flop'] for est in individual_estimates]
    max_flop = max(flop_values)
    min_flop = min(flop_values)
    agreement_ratio = max_flop / min_flop if min_flop > 0 else float('inf')
    
    # All benchmark interpolation uses LOW confidence regardless of agreement
    # due to rapid efficiency improvements making benchmark scaling unreliable
    best_confidence = ConfidenceLevel.LOW
    
    # Create aggregate reasoning
    aggregate_reasoning = f"Multi-benchmark estimation from {len(individual_estimates)} benchmarks: {'; '.join(benchmark_details)} → weighted average: {avg_flop:.2e} FLOP"
    
    return {
        'flop': avg_flop,
        'confidence': best_confidence,
        'method': EstimationMethod.BENCHMARK_BASED,
        'reasoning': aggregate_reasoning,
        'benchmark_count': len(individual_estimates),
        'agreement_ratio': agreement_ratio
    }


def estimate_from_benchmark_score(model: Model, estimator: ComputeEstimator) -> Optional[Dict]:
    """Estimate FLOP from benchmark scores - enhanced multi-benchmark version."""
    # Try new multi-benchmark approach first
    result = estimate_from_multiple_benchmarks(model, estimator)
    if result:
        return result
    
    # Fallback to legacy single-benchmark approach
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
        
        # All benchmark interpolation uses LOW confidence due to inherent unreliability
        result.confidence = ConfidenceLevel.LOW
            
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


def estimate_model_flops(model: Model, estimator: ComputeEstimator, usage_tracker: Optional[MethodUsageTracker] = None, 
                         exclusion_criteria: Optional[DeveloperExclusion] = None) -> Optional[Dict]:
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
        usage_tracker: Optional tracker for method usage statistics
        exclusion_criteria: Optional developer exclusion criteria for applying FLOP caps
        
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
        params, tokens, confidence, matched_pattern = known_specs
        
        # Update model parameters if not already set
        if model.parameters is None:
            model.parameters = params
            model.parameter_source = f"known_specification:{matched_pattern}"
        
        result = estimator.estimate_from_scaling_laws(params, tokens, "chinchilla")
        estimate = {
            'flop': result.flop_estimate,
            'confidence': confidence,
            'method': result.method,
            'reasoning': f"Known model specification '{matched_pattern}': {result.reasoning}",
            'parameters': params,
            'parameter_source': f'known_specification:{matched_pattern}',
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
    
    # Record method usage in tracker
    if usage_tracker and primary_estimate:
        # Map the priority to method ID for tracking
        priority_to_method_id = {
            0: "manual_overrides",
            1: "known_specifications", 
            2: "parameter_based_scaling",
            3: "benchmark_interpolation"
        }
        method_id = priority_to_method_id.get(primary_estimate['priority'], "unknown")
        usage_tracker.record_method_usage(
            method_id=method_id,
            model_name=model.name,
            confidence=primary_estimate['confidence'],
            flop=primary_estimate['flop']
        )
    
    # Infer correct developer from benchmark references if needed
    corrected_developer = infer_developer_from_references(model)
    
    # Apply developer exclusion criteria capping if configured
    if exclusion_criteria and primary_estimate and corrected_developer:
        original_flop = primary_estimate['flop']
        original_confidence = primary_estimate['confidence']
        original_method = primary_estimate['method']
        original_reasoning = primary_estimate['reasoning']
        
        final_flop, final_confidence, final_reasoning, was_capped = exclusion_criteria.apply_resource_cap_if_needed(
            corrected_developer, original_flop, original_confidence, original_method, original_reasoning
        )
        
        if was_capped:
            # Store original estimate as alternative if preserve is enabled
            if exclusion_criteria.should_preserve_original():
                model.add_alternative_estimate(
                    flop=original_flop,
                    confidence=original_confidence,
                    method=original_method,
                    reasoning=f"Original estimate before resource constraint cap: {original_reasoning}"
                )
            
            # Update primary estimate with capped values
            primary_estimate['flop'] = final_flop
            primary_estimate['confidence'] = final_confidence
            primary_estimate['reasoning'] = final_reasoning
    
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
    
    # Initialize configuration and components
    initialize_configuration()
    storage = JSONStorage(args.data_dir)
    estimator = ComputeEstimator()
    usage_tracker = MethodUsageTracker()
    
    # Initialize developer exclusion criteria
    try:
        exclusion_criteria = DeveloperExclusion()
        logger.info(f"Loaded developer exclusion criteria: {exclusion_criteria.get_statistics()}")
    except Exception as e:
        logger.warning(f"Failed to load developer exclusion criteria: {e}")
        logger.warning("Proceeding without developer exclusion filtering")
        exclusion_criteria = None
    
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
            
        # Track total models processed
        usage_tracker.total_models += 1
        
        estimate = estimate_model_flops(model, estimator, usage_tracker, exclusion_criteria)
        if estimate:
            usage_tracker.models_with_estimates += 1
            if args.dry_run:
                logger.info(f"Would update {model.name}: {estimate['flop']:.2e} FLOP "
                          f"(confidence: {estimate['confidence'].value})")
            else:
                # Update the model with status classification
                model.update_flop_estimate(
                    flop=estimate['flop'],
                    confidence=estimate['confidence'],
                    method=estimate['method'],
                    reasoning=estimate['reasoning']
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
        else:
            # No estimate possible
            usage_tracker.record_no_estimate(model.name)
    
    # Status classification is now handled automatically in update_flop_estimate
    
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
    
    # Print enhanced summary with method priority and usage statistics
    print_enhanced_summary(all_models, updated_count, above_threshold_count, usage_tracker, args)


if __name__ == "__main__":
    sys.exit(main())