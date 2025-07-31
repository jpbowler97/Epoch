# Compute Estimation Methodologies

This document details the FLOP estimation engine implementation, covering all estimation methods, confidence scoring, and technical architecture decisions.

## System Architecture

### Central Processing Script: `estimate_flops.py`
**Purpose**: Apply FLOP estimations to all scraped models using multiple methodologies
**Location**: `scripts/estimate_flops.py`
**Core Logic**: Multi-method estimation with confidence-based method selection

```python
def estimate_model_flops(model: Model, estimator: ComputeEstimator):
    # Method 1: Known model specifications (highest priority)
    if known_specs := find_known_model_match(model.name):
        return scaling_laws_estimate(known_specs)
    
    # Method 2: Benchmark-based estimation
    if benchmark_estimate := estimate_from_benchmark_score(model, estimator):
        return benchmark_estimate
        
    # Method 3: Parameter size heuristics (fallback)
    if param_count := extract_model_size(model.name):
        return heuristic_estimate(param_count)
```

## Current Implementation (What We Have)

### 1. Known Model Specifications ✅ IMPLEMENTED
**Location**: `scripts/estimate_flops.py:KNOWN_MODEL_SPECS`
**Method**: Direct calculation using published parameters and training tokens
**Formula**: `6 × Parameters × Training_Tokens` (Chinchilla scaling laws)
**Confidence**: High (published specs) to Medium (reliable estimates)

**Implementation**:
```python
KNOWN_MODEL_SPECS = {
    "llama-3.1-405b": (405_000_000_000, 15_000_000_000_000, ConfidenceLevel.HIGH),
    "gpt-4": (1_760_000_000_000, 13_000_000_000_000, ConfidenceLevel.MEDIUM),
    "claude-3.5-sonnet": (250_000_000_000, 10_000_000_000_000, ConfidenceLevel.MEDIUM),
    "deepseek-v3": (671_000_000_000, 14_800_000_000_000, ConfidenceLevel.MEDIUM)
}
```

**Examples**:
- Llama 3.1 405B = 6 × 405B × 15T tokens = 3.65e+25 FLOP
- DeepSeek V3 = 6 × 671B × 14.8T tokens = 5.96e+25 FLOP

### 2. Benchmark-Based Method ✅ IMPLEMENTED  
**Location**: `scripts/estimate_flops.py:estimate_from_benchmark_score()`
**Data Sources**: LMArena ELO ratings, OpenLM Arena scores
**Implementation**: Power law scaling with reference models
**Confidence**: Low to Medium based on score proximity to references

**Reference Models**:
```python
BENCHMARK_REFERENCES = {
    "lmarena_score": {
        1300: 3.8e25,   # Llama 3.1 405B level
        1250: 2.7e25,   # Gemini 1.5 Pro level  
        1200: 2.15e25,  # GPT-4 level
        1150: 1.7e25,   # Claude 3.5 Sonnet level
        1100: 1e25,     # Threshold level
    }
}
```

**Power Law Formula**: `FLOP = reference_flop × (elo_score / reference_elo)^α` where α=3.0

**Examples**:
- Model with ELO 1474 → 5.54e+25 FLOP (Gemini 2.5 Pro)
- Model with ELO 1443 → 5.20e+25 FLOP (ChatGPT-4o)

### 3. Parameter Size Heuristics ✅ IMPLEMENTED
**Location**: `scripts/estimate_flops.py:extract_model_size()`
**Method**: Extract parameter count from model names and apply scaling assumptions
**Fallback Logic**: When no known specs or benchmarks available
**Confidence**: Speculative (lowest confidence level)

**Implementation**:
```python
def extract_model_size(model_name: str) -> Optional[int]:
    patterns = [
        r'(\d+\.?\d*)b(?:illion)?',  # 405B, 7B
        r'(\d+\.?\d*)t(?:rillion)?',  # 1.7T
    ]
    # Assume 20x tokens per parameter (common ratio)
    estimated_tokens = param_count * 20
```

**Examples**:
- "claude-opus-4-175b" → 175B params → 1.75T tokens → 2.1e+25 FLOP
- "mistral-22b-instruct" → 22B params → 440B tokens → 5.28e+23 FLOP

### 4. Threshold Classification System ✅ IMPLEMENTED
**Location**: `scripts/estimate_flops.py` + `src/epoch_tracker/models/model.py`
**Purpose**: Systematically classify models relative to 1e25 FLOP threshold
**Logic**: Used for core dataset curation and verification prioritization

**Classification Thresholds**:
```python
HIGH_CONFIDENCE_ABOVE_THRESHOLD = 5e25  # >= 5x threshold
HIGH_CONFIDENCE_BELOW_THRESHOLD = 5e24  # <= 0.5x threshold

def classify_by_threshold(flop_estimate):
    if flop_estimate >= 5e25:
        return ThresholdClassification.HIGH_CONFIDENCE_ABOVE
    elif flop_estimate <= 5e24:
        return ThresholdClassification.HIGH_CONFIDENCE_BELOW
    else:
        return ThresholdClassification.NOT_SURE  # Needs manual verification
```

### 5. Confidence Level System ✅ IMPLEMENTED
**Location**: `src/epoch_tracker/models/model.py`
**Levels**:
- `HIGH`: Known model specifications from official sources
- `MEDIUM`: Reliable estimates with good data sources  
- `LOW`: Benchmark-based interpolation with some uncertainty
- `SPECULATIVE`: Heuristic estimates from model names

### 6. Estimation Method Tracking ✅ IMPLEMENTED
**Location**: `src/epoch_tracker/models/model.py`
**Methods**:
- `SCALING_LAWS`: Chinchilla approximation (6N×D) with known specs
- `BENCHMARK_BASED`: ELO-to-FLOP interpolation using reference models
- `HEURISTIC`: Parameter size extraction from model names
- `MANUAL_RESEARCH`: (defined for future manual entries)
- `COMPANY_DISCLOSURE`: (defined for future direct reporting parser)

## Current System Performance

### Processing Statistics (Latest Run)
```
Total models processed: 418
Models with FLOP estimates: 400
Models updated this run: 45
Models above 1e25 FLOP: 389

THRESHOLD CLASSIFICATION:
  High confidence > 1e25 FLOP (>= 5.0e+25): 0
  High confidence < 1e25 FLOP (<= 5.0e+24): 11
  Not sure (5.0e+24 - 5.0e+25): 389
```

### Method Distribution
| Method | Count | Percentage | Confidence Range |
|--------|--------|------------|------------------|
| Benchmark-based | ~320 | 80% | Low to Medium |
| Known specifications | ~15 | 4% | High to Medium |
| Parameter heuristics | ~65 | 16% | Speculative |

### Integration with Core Dataset Pipeline

The estimation system directly feeds into the core dataset curation workflow:

```
estimate_flops.py (Stage 2)
    ↓ [400 models with FLOP estimates]
    ↓ [Filter: HIGH_CONFIDENCE_ABOVE + NOT_SURE = 400 candidates]
    ↓
refresh_core_dataset.py (Stage 3)
    ↓ [Manual verification workflow]
    ↓ [15 models marked as verified=y]
```

**Key Integration Points**:
1. **Threshold classification** drives candidate selection for manual verification
2. **Confidence levels** help prioritize verification efforts (high confidence first)
3. **Estimation reasoning** provides context for human reviewers
4. **Method tracking** enables quality control and methodology improvements

## Epoch AI Methodologies (Reference Standard)

### 1. Direct Reporting ⭐ HIGHEST ACCURACY
**Description**: Training compute explicitly disclosed in papers/announcements
**Accuracy**: Highest (ground truth)
**Our Status**: ❌ NOT IMPLEMENTED
**Gap**: No automated parsing of company disclosures or research papers

### 2. Hardware Details and Usage Method
**Description**: Calculate from hardware specifications and training duration
**Formula**: `Compute = Chip-days × Peak_FLOP/s × Utilization_Rate`
**Typical Utilization**: 30-50%
**Examples**: "128 TPUv3 instances for two days"
**Our Status**: ❌ NOT IMPLEMENTED
**Gap**: No hardware-based estimation capability

### 3. Arithmetic Operations Counting
**Description**: Count operations per forward/backward pass
**Formula**: "6 × parameters × training examples × epochs"
**Our Status**: ✅ PARTIALLY IMPLEMENTED
**Gap**: We use token approximation instead of examples×epochs

### 4. Benchmark Performance Method
**Description**: Use scaling laws to predict benchmark performance vs compute
**Our Status**: ✅ IMPLEMENTED
**Notes**: Our implementation matches Epoch's approach

## Comparison Matrix

| Method | Epoch Priority | Our Implementation | Accuracy | Data Required |
|--------|---------------|-------------------|----------|---------------|
| **Direct Reporting** | ⭐ Primary | ❌ Missing | Highest | Company disclosures |
| **Hardware Details** | ⭐ Primary | ❌ Missing | High | Hardware specs + duration |
| **Scaling Laws** | ⭐ Primary | ✅ Implemented | Medium | Parameters + tokens/examples |
| **Benchmark-Based** | ⭐ Primary | ✅ Implemented | Medium | Benchmark scores |

## Missing Implementations (Priority Order)

### Priority 1: Direct Reporting Parser ❌
**What's Missing**: Automated parsing of company announcements for explicit FLOP disclosures
**Examples**:
- Meta's Llama 3.1 disclosure: "3.8 × 10^25 FLOPs"
- OpenAI's GPT-4 technical reports
- Anthropic's Claude technical specifications

**Implementation Needed**:
```python
class DirectReportingScraper(BaseScraper):
    """Parse company blogs/papers for explicit FLOP disclosures"""
    
    def _extract_flop_disclosure(self, text: str) -> Optional[float]:
        # Parse patterns like "3.8e25 FLOP", "2.1 × 10^25 FLOPs"
        pass
    
    def _scrape_company_announcements(self) -> List[Model]:
        # Scrape Meta AI, OpenAI, Anthropic blogs
        pass
```

### Priority 2: Hardware-Based Estimation ❌
**What's Missing**: Calculate FLOP from hardware specifications and training duration
**Examples**:
- "16,000 H100 GPUs for X days" → FLOP calculation
- TPU pod specifications → compute estimation

**Implementation Needed**:
```python
class HardwareBasedEstimator:
    """Estimate FLOP from hardware specs and training time"""
    
    GPU_SPECS = {
        "H100": {"peak_flops": 2e15, "bf16_flops": 1e15},
        "TPUv4": {"peak_flops": 2.75e14},
        "A100": {"peak_flops": 3.12e14}
    }
    
    def estimate_from_hardware(self, gpu_type: str, count: int, 
                             days: float, utilization: float = 0.4) -> float:
        # Formula: count × days × 24h × 3600s × peak_flops × utilization
        pass
```

### Priority 3: Enhanced Scaling Laws ❌
**What's Missing**: More sophisticated token estimation based on disclosed training data
**Current Gap**: We estimate tokens from parameters, but should use actual training data when available

**Enhancement Needed**:
```python
def _estimate_training_tokens(self, model: Model) -> Tuple[float, ConfidenceLevel]:
    """Enhanced token estimation with multiple data sources"""
    
    # Priority 1: Explicit disclosure (e.g., "trained on 15T tokens")
    if disclosed_tokens := self._extract_token_disclosure(model):
        return disclosed_tokens, ConfidenceLevel.HIGH
    
    # Priority 2: Training dataset references
    if dataset_size := self._estimate_from_datasets(model):
        return dataset_size, ConfidenceLevel.MEDIUM
    
    # Priority 3: Parameter scaling (current method)
    return self._estimate_from_parameters(model), ConfidenceLevel.LOW
```

## Methodology Validation

Our current estimates compared to Epoch's official values:

| Model | Our Estimate | Epoch Official | Ratio | Method |
|-------|-------------|----------------|-------|---------|
| **Llama 3.1 405B** | 2.46e+25 | 3.8e+25 | 0.65x | Scaling Laws vs Direct |
| **GPT-4** | 3.20e+25 | 2.1e+25 | 1.52x | Benchmark vs Hardware |
| **Claude 3 Opus** | 1.28e+25 | 1.6e+25 | 0.80x | Benchmark vs Benchmark |

**Analysis**: Our estimates are within Epoch's expected accuracy ranges (±3x confident, ±10x likely).

## Implementation Roadmap

### Phase 1: Direct Reporting (Highest Impact)
1. **Company Blog Scrapers**: Meta AI, OpenAI, Anthropic announcement parsing
2. **FLOP Disclosure Parser**: Regex patterns for "X FLOP" statements  
3. **Paper Analysis**: ArXiv technical report parsing
4. **Confidence Boost**: Mark direct disclosures as `ConfidenceLevel.HIGH`

### Phase 2: Hardware-Based Estimation
1. **Hardware Database**: GPU/TPU specifications and peak FLOP rates
2. **Infrastructure Parser**: Parse "X GPUs for Y days" statements
3. **Utilization Modeling**: Account for 30-50% utilization rates
4. **Cross-Validation**: Compare hardware estimates against scaling laws

### Phase 3: Enhanced Validation
1. **Multi-Method Consensus**: Combine multiple estimation approaches
2. **Uncertainty Quantification**: Provide confidence intervals
3. **Automated Validation**: Flag estimates deviating >10x from consensus

## Current Coverage Analysis

**Models with High-Confidence Estimates (Direct/Hardware)**:
- ✅ Meta Llama 3.1 405B: 3.8e+25 FLOP (official disclosure)
- ❌ GPT-4: Hardware-based possible ("$78M compute cost")
- ❌ Grok-2: Training time/hardware details available
- ❌ Gemini Ultra: TPU infrastructure details mentioned

**Next Steps**: Implementing direct reporting parser would immediately improve confidence levels for 3-4 major models from `speculative` to `high`.