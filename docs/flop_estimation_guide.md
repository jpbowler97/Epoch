# FLOP Estimation Guide

This guide explains how the Epoch AI Model Tracker estimates training FLOP for AI models and assigns confidence scores.

## Overview

The system automatically estimates training FLOP (floating-point operations) for models scraped from various sources. The goal is to identify models likely trained with over 1e25 FLOP using multiple estimation methods with confidence scoring.

## Estimation Pipeline

### 1. Data Collection
Models are collected from sources like:
- **LMArena**: Model rankings with ELO scores
- **Papers with Code**: Academic releases
- **Manual datasets**: Curated CSV files

### 2. FLOP Estimation Process

The `estimate_flops.py` script applies estimations using this hierarchy:

```bash
python scripts/estimate_flops.py --update
```

#### Method Priority (Best to Worst):

0. **Manual Overrides** (Confidence: High/Medium/Low) - **HIGHEST PRIORITY**
1. **Known Model Specifications** (Confidence: High/Medium) - **PREFERRED**
2. **Parameter-based Chinchilla Scaling** (Confidence: Medium/Low) - **RECOMMENDED** 
3. **Benchmark Score Interpolation** (Confidence: Medium/Low) - **FALLBACK**  
4. **Parameter Size Heuristics** (Confidence: Speculative) - **DEPRECATED**

## Estimation Methods

### 0. Manual Overrides (NEW)

**How it works:**
- Uses curated FLOP values from Epoch AI's research tracker
- Takes absolute highest priority over all other methods
- Provides authoritative estimates for frontier models

**Source:** https://epoch.ai/data-insights/models-over-1e25-flop

**Examples:**
```python
MANUAL_OVERRIDES = {
    # High-precision estimates (HIGH confidence)
    "llama_3.1_405b": (3.8e25, ConfidenceLevel.HIGH, "Epoch AI: High-precision estimate from Meta disclosure"),
    "grok_2": (3.0e25, ConfidenceLevel.HIGH, "Epoch AI: High-precision estimate from xAI disclosure"),
    
    # Low-precision but reliable estimates (MEDIUM confidence)
    "claude_3_opus": (1.6e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from industry analysis"),
    "gpt_4": (2.1e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from scaling analysis"),
    "claude_3.5_sonnet": (3.6e25, ConfidenceLevel.MEDIUM, "Epoch AI: Low-precision estimate from benchmarks"),
    
    # Speculative but notable estimates (LOW confidence)
    "claude_opus_4": (1.5e26, ConfidenceLevel.LOW, "Epoch AI: Speculative estimate for next-gen Claude"),
    "gpt_4.5": (6.4e25, ConfidenceLevel.LOW, "Epoch AI: Low-precision estimate for GPT-4.5"),
}
```

**Key Benefits:**
- **Authoritative data**: Based on Epoch's expert analysis and research
- **Handles edge cases**: Covers frontier models that other methods struggle with  
- **Correct Claude estimates**: Fixes previous issues with Claude 3 Opus and Claude 4 Opus
- **Transparent sourcing**: All values include reasoning and confidence levels

**Confidence:** High (official disclosures) → Medium (industry analysis) → Low (speculative estimates)

### 1. Known Model Specifications

**How it works:**
- Matches model names against a database of known specifications
- Uses published parameter counts and training token estimates
- Applies Chinchilla scaling law: `FLOP = 6 × Parameters × Tokens`

**Example:**
```python
# llama-3.1-405b matches known spec:
params = 405_000_000_000      # 405B parameters
tokens = 15_000_000_000_000   # 15T tokens  
flop = 6 * params * tokens    # = 3.65e25 FLOP
confidence = "high"           # Published specification
```

**Confidence:** High (based on official disclosures)

### 2. Parameter-based Chinchilla Scaling

**How it works:**
- Extracts parameter count from model names (e.g., "70b", "32b", "405b")
- Applies intelligent token estimation based on model era and characteristics
- Uses Chinchilla scaling law: `FLOP = 6 × Parameters × Tokens`

**Token Estimation Logic:**
```python
# Modern era models (2023+) - Llama 3+, Qwen 3, Gemma 3
- Large models (≥100B): 15 tokens/param (conservative)
- Smaller models: 20 tokens/param (higher ratio)

# Mid-era models (2022-2023) - Llama 2, Qwen 2, Mistral  
- All sizes: 12 tokens/param (moderate ratios)

# Specialized models - Code/instruct variants
- All sizes: 18 tokens/param (higher training ratios)

# Chinese models - Qwen, GLM, ChatGLM, Yi
- All sizes: 16 tokens/param (well-trained models)

# Default fallback
- All sizes: 15 tokens/param (conservative estimate)
```

**Examples:**
```python
# qwen3_32b → 32B params, modern model
tokens = 32_000_000_000 * 20 = 640B tokens
flop = 6 * 32B * 640B = 1.23e+23 FLOP (Medium confidence)

# llama2_70b_steerlm → 70B params, mid-era model
tokens = 70_000_000_000 * 12 = 840B tokens  
flop = 6 * 70B * 840B = 3.53e+23 FLOP (Medium confidence)

# gemma_7b → 7B params, generic estimate
tokens = 7_000_000_000 * 15 = 105B tokens
flop = 6 * 7B * 105B = 4.41e+21 FLOP (Low confidence)
```

**Confidence:** Medium (modern/specialized models) → Low (generic/early models)

### 3. Multi-Benchmark Score Estimation (NEW)

**How it works:**
- Uses multiple benchmark scores from the same model (OpenLM Arena ELO, coding scores, reasoning scores, etc.)
- Each benchmark type has its own reference model and scaling parameters
- Aggregates estimates using confidence-weighted averaging
- Boosts confidence when multiple benchmarks agree

**Benchmark Reference Models:**
```python
BENCHMARK_REFERENCE_MODELS = {
    "lmarena_score": {
        "reference_model": "llama_3.1_405b",
        "reference_score": 1335.0,    # Actual LMArena score
        "reference_flop": 3.8e25,     # Epoch AI research
        "scaling_exponent": 3.0       # Power law α
    },
    "openlm_arena_elo": {
        "reference_model": "llama_3.1_405b", 
        "reference_score": 1286.0,    # Actual OpenLM Arena ELO
        "reference_flop": 3.8e25,
        "scaling_exponent": 3.0
    },
    "coding_score": {
        "reference_model": "llama_3.1_405b",
        "reference_score": 1299.0,    # Coding performance
        "reference_flop": 3.8e25,
        "scaling_exponent": 3.0
    },
    "aai_score": {
        "reference_model": "llama_3.1_405b",
        "reference_score": 40.5,      # AI safety score  
        "reference_flop": 3.8e25,
        "scaling_exponent": 2.0       # More conservative scaling
    },
    "mmlu_pro_score": {
        "reference_model": "llama_3.1_405b",
        "reference_score": 73.2,      # Advanced reasoning
        "reference_flop": 3.8e25,
        "scaling_exponent": 2.5       # Moderate scaling
    }
}
```

**Multi-Benchmark Aggregation:**
1. **Individual estimates**: Each available benchmark generates a FLOP estimate
2. **Confidence weighting**: Higher confidence benchmarks get more weight
3. **Agreement checking**: If estimates agree (within 2x), confidence is boosted
4. **Weighted average**: Final estimate combines all benchmarks

**Example:**
```python
# Model has multiple benchmark scores:
benchmarks = {
    "openlm_arena_elo": 1350,  # → 4.2e25 FLOP (Medium confidence)
    "coding_score": 1320,      # → 3.9e25 FLOP (Medium confidence)  
    "mmlu_pro_score": 75.0     # → 4.1e25 FLOP (Medium confidence)
}

# Estimates agree (within 2x) → confidence boosted to Medium
# Weighted average: ~4.1e25 FLOP (Medium confidence)
```

**Key Benefits:**
- **Higher accuracy**: Multiple data points reduce estimation errors
- **Improved confidence**: Agreement across benchmarks validates estimates
- **Domain-specific scaling**: Different α values for different benchmark types
- **Extensible design**: Easy to add new benchmarks

**Confidence:** Medium (multiple agreeing benchmarks) → Low (single benchmark or disagreement)

### 4. Parameter Size Heuristics (DEPRECATED)

**Note:** This method has been largely replaced by Parameter-based Chinchilla Scaling (Method 2) which provides more intelligent token estimation.

**How it works:**
- Extracts parameter count from model names (e.g., "70B", "405B")  
- Uses generic 20:1 token-to-parameter ratio
- Applies Chinchilla scaling laws

**Example:**
```python
# Generic fallback only used when other methods fail
params = extracted_param_count
tokens = params * 20          # Fixed 20:1 ratio
flop = 6 * params * tokens    # Chinchilla scaling
confidence = "speculative"    # Low confidence
```

**Confidence:** Speculative (replaced by Method 2)

## Confidence Levels

| Level | Description | Typical Source |
|-------|-------------|----------------|
| **High** | Official specifications, published papers | Company disclosures, research papers |
| **Medium** | Reliable estimates from benchmarks/interpolation | Close benchmark matches, industry estimates |
| **Low** | Distant interpolation, older estimates | Distant benchmark interpolation |
| **Speculative** | Heuristic-based rough estimates | Name parsing, assumed ratios |

## Model Status Classification

Based on FLOP estimates and confidence, models are classified:

- **Confirmed Above**: High confidence ≥ 1e25 FLOP
- **Likely Above**: Medium/Low confidence ≥ 1e25 FLOP  
- **Likely Below**: Medium/Low confidence < 1e25 FLOP
- **Confirmed Below**: High confidence < 1e25 FLOP
- **Uncertain**: No reliable FLOP estimate

## Usage Examples

### Apply FLOP Estimations

```bash
# Dry run to see what would be estimated
python scripts/estimate_flops.py --dry-run --verbose

# Apply estimations to all models
python scripts/estimate_flops.py --update

# View results
python scripts/query_models.py --above-threshold --format table
```

### Query by Confidence

```bash
# Only high-confidence models
python scripts/query_models.py --confidence high --format table

# All models above threshold
python scripts/query_models.py --above-threshold --sort-by flop
```

## Adding New Models

### Adding Manual Overrides (Recommended for Frontier Models)

To add authoritative FLOP estimates for frontier models, update `MANUAL_OVERRIDES` in `scripts/estimate_flops.py` (lines 31-53):

**IMPORTANT:** Use exact FLOP values from Epoch AI's tracker or other authoritative sources.

```python
MANUAL_OVERRIDES = {
    # Pattern format: "normalized_model_name": (flop_value, confidence_level, reasoning)
    "new_frontier_model": (4.2e25, ConfidenceLevel.MEDIUM, "Epoch AI: Description of source"),
}
```

**Guidelines:**
- **Use normalized names**: Must match scraped data format (underscores, not hyphens)
- **Confidence levels**: 
  - HIGH: Official company disclosures, published papers
  - MEDIUM: Industry analysis, reliable third-party estimates  
  - LOW: Speculative estimates, unconfirmed rumors
- **Include source**: Always reference where the estimate came from
- **Check Epoch's tracker**: Sync with https://epoch.ai/data-insights/models-over-1e25-flop

### Adding New Benchmark References

The system uses a **JSON configuration file** for seamless addition of new benchmark references. No code changes required!

**🚀 Quick Start:**
1. Edit `configs/benchmark_references.json`
2. Add your benchmark data
3. Run estimation - everything works automatically

**Example JSON Entry:**
```json
{
  "benchmark_references": {
    "your_new_benchmark": {
      "reference_model": "llama_3.1_405b",
      "reference_score": 85.4,
      "reference_flop": 3.8e25,
      "source": "Your Benchmark + Epoch AI disclosure analysis",
      "source_url": "https://your-benchmark.com/",
      "scaling_exponent": 2.5,
      "benchmark_type": "percentage",
      "notes": "Description of benchmark"
    }
  }
}
```

**Field Guidelines:**
- **`reference_model`**: Usually "llama_3.1_405b" (well-characterized)
- **`reference_score`**: Actual benchmark score for reference model
- **`reference_flop`**: Training FLOP of reference model (3.8e25 for Llama 405B)
- **`scaling_exponent`**: 3.0 (ELO scores), 2.5 (reasoning), 2.0 (safety/specialized)
- **`source`**: Citation for benchmark score + FLOP estimate

**Examples:**

**Vision Benchmark:**
```json
"vision_score": {
  "reference_model": "gpt_4o",
  "reference_score": 88.5,
  "reference_flop": 3.8e25,
  "source": "Vision Benchmark + Epoch AI estimates",
  "scaling_exponent": 2.5
}
```

**Math Reasoning:**
```json
"math_reasoning": {
  "reference_model": "llama_3.1_405b",
  "reference_score": 76.8,
  "reference_flop": 3.8e25,
  "source": "Math Benchmark + Epoch AI disclosure",
  "scaling_exponent": 2.5
}
```

**📖 Complete Guide:** See [Adding Benchmark References](adding_benchmark_references.md) for detailed workflow, troubleshooting, and examples.

### Adding New Reference Models

To improve estimation accuracy, add known models to `KNOWN_MODEL_SPECS` in `scripts/estimate_flops.py`:

**IMPORTANT:** Use normalized model names (underscores, not hyphens) to match the data pipeline.

```python
KNOWN_MODEL_SPECS = {
    # Must match normalized names from scraped data (e.g., "llama_3.1_405b" not "llama-3.1-405b")
    "normalized_model_name": (
        parameters,        # int: parameter count (e.g., 405_000_000_000)
        training_tokens,   # int: training tokens (e.g., 15_000_000_000_000)
        confidence_level   # ConfidenceLevel.HIGH/MEDIUM/LOW/SPECULATIVE
    ),
}
```

**Examples of good additions:**
- Official model releases with disclosed specifications → HIGH confidence
- Industry estimates from reliable sources → MEDIUM confidence
- Research paper estimates → MEDIUM confidence
- Speculative estimates from limited data → LOW confidence

**Check normalized names:** Look at `data/scraped/lmarena_models.json` to see actual model names after normalization.

## Validation and Quality Control

### Check Estimation Quality
```bash
# Compare against known models
python scripts/validate_against_epoch.py

# Show estimation statistics
python scripts/query_models.py --stats
```

### Common Issues

1. **Model name parsing failures**: Add patterns to `extract_model_size()`
2. **Missing reference points**: Add benchmark scores to `BENCHMARK_REFERENCES`
3. **Outdated specifications**: Update `KNOWN_MODEL_SPECS` with latest data

## Implementation Details

### Key Files
- `scripts/estimate_flops.py`: Main estimation pipeline
- `src/epoch_tracker/estimation/compute_estimator.py`: Core estimation methods
- `src/epoch_tracker/models/model.py`: Model schema and confidence levels

### Extending the System

To add new estimation methods:

1. Add method to `ComputeEstimator` class
2. Update `estimate_model_flops()` function in `estimate_flops.py`
3. Add appropriate confidence level mapping
4. Test with known models for validation

## Performance Metrics

Current system performance (as of last update):
- **Total models processed**: 217
- **Models with FLOP estimates**: 134 (62%)
- **Models above 1e25 FLOP**: 27 (12%)
- **High confidence estimates**: ~15% of total
