# AI Models Above 1e25 FLOP: Documentation

## Executive Summary

This document describes the methodology used to produce the table in [`data/staging/above_1e25_flop_staging.csv`](../data/staging/above_1e25_flop_staging.csv). It covers how we automatically identify candidate models, estimate their training FLOPs (floating-point operations) and then manually verify to add to our dataset to ensure our list is accurate and comprehensive.

**For proposed changes to the existing Epoch AI Models Documentation, see: [`proposed_documentation_changes.md`](proposed_documentation_changes.md)**

## 1. Inclusion Process

### 1.1 Overview

The inclusion process follows a four-stage pipeline designed to transform raw model information from multiple sources into a high-quality, verified dataset:

```
Stage 1: Identify New Models → Stage 2: Estimate FLOPs → Stage 3: Manual Review
```

Each stage operates independently and maintains complete data lineage for auditability.

### 1.2 Stage 1: Identify New Models

#### 1.2.1 Data Sources

The system aggregates model information from multiple sources using a configurable scraper architecture:

**Automated Web Scrapers:**
- **LMArena** (lmarena.ai): Model rankings with ELO scores, updated weekly
- **OpenLM Arena** (openlm.ai): Chatbot performance leaderboards, real-time updates
- **Papers with Code**: Academic model releases (limited coverage)

**Browser-Based Manual Collection** (Sites requiring interactive browser access):
- **SuperCLUE**: Chinese language understanding benchmark
- **Physics-IQ**: Video model physics reasoning evaluation  
- **Olympic Arena**: Multi-discipline cognitive assessment
- **Video Arena**: Text-to-video model quality rankings

These sites require actual browser interaction (clicking through pages, waiting for dynamic content, handling interactive elements) and cannot be accessed via conventional HTTP requests. Data from these sources is collected either manually or using an AI agent with browser access.


#### 1.2.2 Collection Methodology

The collection process ensures comprehensive coverage and data quality through:

**Data Processing:**
- Model names are standardized to enable consistent tracking across sources
- Duplicate models are identified and merged using model name and developer information
- Each data point maintains full provenance including source URL and collection timestamp

**Quality Control:**
- Generic or incomplete model names are filtered out (e.g., "gpt" or "claude" without version numbers)
- Test and demo models are excluded from the dataset
- Models are validated against known naming patterns and developer information

### 1.3 Stage 2: FLOP Estimation

The system applies a hierarchical series of estimation methods to calculate training FLOP for each model. Methods are attempted in priority order, with the first successful method used.

#### 1.3.1 Estimation Hierarchy

0. **Manual Overrides** (Highest Priority)
   - Source: Authoritative estimates from expert analysis and industry research
   - Coverage: ~30 frontier models with well-established compute requirements
   - Confidence: HIGH (manual overrides represent our highest confidence estimates)

1. **Chinchilla Scaling** 
   - Source: Published papers, official disclosures, or model names containing parameter counts
   - Method: Apply the Chinchilla scaling law: 6 × parameters × tokens
   - Justification: The factor of 6 comes from the forward pass (2N operations) plus backward pass (4N operations) for N parameters. See [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) and [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)
   - Confidence: MEDIUM (reliable when parameter and token counts are known or can be reasonably estimated)

2. **Benchmark Interpolation**
   - Source: Performance scores (ELO, MMLU, coding benchmarks)
   - Method: Power law scaling from reference models with known FLOP
   - Confidence: LOW (inherently unreliable due to rapid efficiency improvements)
   - **Important caveat**: This method becomes increasingly unreliable over time as newer models achieve similar benchmark scores with less compute through distillation, improved architectures, and better training techniques

3. **Speculative Estimates**
   - Source: Indirect indicators such as developer statements, hardware availability, or rough comparisons
   - Method: Order-of-magnitude estimates based on limited information
   - Confidence: SPECULATIVE (used only when no other methods are available)
   - These estimates may be off by an order of magnitude and are included primarily for completeness

#### Summary of Confidence Levels

| Confidence Level | Estimation Methods | Reliability | Typical Error Range |
|---|---|---|---|
| **HIGH** | Manual overrides from authoritative sources | Most reliable | Within 50% |
| **MEDIUM** | Chinchilla scaling with known or estimated parameters | Reliable | Within 2-3× |
| **LOW** | Benchmark interpolation | Unreliable | Within 5-10× |
| **SPECULATIVE** | Indirect estimates, limited information | Highly uncertain | Order of magnitude |

#### 1.3.2 Developer-Based Exclusion Criteria

Some developers are known to lack the computational resources (either financially or due to limited GPU access) to train models exceeding 10²⁵ FLOP. To prevent overestimation, models from these developers are automatically capped at 9.9×10²⁴ FLOP.

**Currently affected developers:**
- Mistral AI (limited computational resources)
- Microsoft (Azure-hosted models without clear compute disclosure)
- Additional developers as identified through industry analysis

**Maintenance:** This list is manually maintained in `configs/developer_exclusion_criteria.json` and updated based on:
- Known computational budgets and infrastructure
- Historical training patterns
- Industry intelligence about available resources

### 1.4 Stage 3: Manual Review

After FLOP estimation, all models are classified based on their estimates and sent for appropriate handling:

**Models sent for manual review:**
- **confirmed_above_1e25**: FLOP ≥ 10²⁵ with HIGH confidence → Sent for verification
- **likely_above_1e25**: FLOP ≥ 10²⁵ with MEDIUM or LOW confidence → Sent for verification
- **uncertain**: Models where the estimation method cannot reliably determine if they exceed 10²⁵ FLOP → Sent for verification

**Models excluded from review:**
- **likely_below_1e25**: FLOP < 10²⁵ with MEDIUM or LOW confidence → Excluded
- **confirmed_below_1e25**: FLOP < 10²⁵ with HIGH confidence → Excluded

All three categories sent for manual review (confirmed_above, likely_above, and uncertain) are handled identically in the review process. The classification helps track the reliability of our estimates, but every model potentially at or above the threshold undergoes human verification to ensure accuracy.

### 1.5 Manual Review Process

Each candidate model undergoes manual review where human experts verify the FLOP estimates and make final inclusion decisions.

**Review Process:**
During manual review, each model is examined with its complete metadata, FLOP estimates, alternative estimation methods, and source documentation. Reviewers verify the estimates against available evidence and make a determination about whether the model truly exceeds the 10²⁵ FLOP threshold.

**Verification Protection:**
Once a model has been manually reviewed and marked as verified, this decision is preserved even when the automated pipeline is rerun with updated data or methods. This ensures that human expertise and judgments are not inadvertently overwritten by automated processes.

#### Quality Assurance

Each verification decision requires:
- Documented reasoning in notes field
- Cross-reference of multiple sources when available
- Consistency check against known model families
- Review of estimation method appropriateness

### 1.6 Final Dataset

Verified models are stored in `data/staging/above_1e25_flop_staging.csv` with full metadata, FLOP estimates, and audit trail documenting the verification process.

## 2. Schema Definitions

The staging dataset (`data/staging/above_1e25_flop_staging.csv`) contains the following fields:

| Field | Type | Definition | Example | Population Method |
|---|---|---|---|---|
| **model** | string | Normalized model name in snake_case format | `gpt_4o`, `claude_3_opus` | Automatic (scraping + normalization) |
| **developer** | string | Organization that developed the model | `OpenAI`, `Anthropic` | Automatic (source data) |
| **release_date** | date | First public announcement or API availability | `2024-05-13` | Automatic/Manual |
| **parameters** | float | Total number of model parameters | `1.76e+12` (1.76T) | Automatic (when disclosed) |
| **parameter_source** | string | Source and method for parameter count | `known_specification:gpt_4` | Automatic |
| **training_flop** | float | Estimated training compute in FLOP | `2.1e+25` | Automatic (estimation pipeline) |
| **confidence** | enum | Confidence level of FLOP estimate | `high`, `medium`, `low`, `speculative` | Automatic |
| **confidence_explanation** | string | Detailed reasoning for confidence assignment | `Official disclosure from Meta` | Automatic |
| **estimation_method** | enum | Primary method used for FLOP calculation | `epoch_estimate`, `scaling_laws`, `benchmark_based` | Automatic |
| **alternative_methods** | string | Other estimation attempts and results | `Scaling Laws: 8.4e+24 (Medium)` | Automatic |
| **threshold_classification** | enum | Relationship to 10²⁵ FLOP threshold | `high_confidence_above_1e25` | Automatic |
| **status** | enum | Overall inclusion status | `confirmed_above_1e25`, `uncertain` | Automatic/Manual |
| **reasoning** | text | Detailed calculation or estimation logic | `Chinchilla scaling: 6 × 405B params × 15T tokens` | Automatic |
| **sources** | text | URLs and references for data | `https://arxiv.org/abs/2407.21783` | Automatic |
| **notes** | text | Manual annotations and verification reasoning | `Verified via official Meta blog post` | Manual |
| **verified** | boolean | Manual verification flag (y/n) | `y` | Manual (review process) |
| **last_updated** | timestamp | Last modification timestamp | `2025-02-01T12:00:00Z` | Automatic |
| **blacklist_status** | string | Developer exclusion flag | `capped_insufficient_resources` | Automatic |
| **original_estimate** | float | Initial FLOP estimate before adjustments | `3.0e+25` | Automatic |

### Potential Additional Fields

Based on the published dataset schema, the following fields could potentially be added through automation:

| Field | Feasibility | Implementation Method |
|---|---|---|
| **Domain** | High | Extract from benchmark types (language, vision, multimodal) |
| **Task** | Medium | Infer from benchmark participation |
| **Country** | High | Map from developer organization |
| **Base model** | Medium | Parse from model names (e.g., `_instruct`, `_chat` suffixes) |
| **Hardware quantity** | Low | Requires additional disclosure tracking |

## 3. FLOP Estimation Methodology

### 3.1 Theoretical Foundation

Training FLOP estimation relies on the Chinchilla scaling law, which establishes that optimal training requires approximately 6 floating-point operations per parameter per token:

```
Training FLOP = 6 × N_parameters × N_tokens
```

The factor of 6 comes from the forward pass (2N operations) plus backward pass (4N operations) for N parameters. This relationship holds across model scales from 70M to 500B+ parameters. For detailed explanation, see [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) and the [scaling laws overview](https://www.gwern.net/Scaling-hypothesis#chinchilla-scaling-laws).

### 3.2 Estimation Methods

#### 3.2.1 Method 0: Manual Overrides

**Description**: Authoritative FLOP values from Epoch AI's research team, based on comprehensive analysis of disclosures, leaks, and industry intelligence.

**Implementation**:
```python
MANUAL_OVERRIDES = {
    "gpt_4": (2.1e25, HIGH, "Epoch AI: OpenAI disclosure analysis"),
    "claude_3_opus": (1.6e25, MEDIUM, "Epoch AI: Industry estimates"),
    "llama_3.1_405b": (3.8e25, HIGH, "Epoch AI: Meta paper")
}
```

**Confidence Assignment**:
- HIGH: Official company disclosures, published papers
- MEDIUM: Reliable industry analysis, consistent estimates
- LOW: Speculative or conflicting reports

#### 3.2.2 Method 1: Known Model Specifications

**Description**: Direct application of Chinchilla scaling to models with published parameter counts and training token counts.

**Example Calculation**:
```
Model: Llama 3.1 405B
Parameters: 405,000,000,000
Training tokens: 15,000,000,000,000
FLOP = 6 × 4.05e11 × 1.5e13 = 3.65e25
Confidence: HIGH (published specification)
```

#### 3.2.3 Method 2: Parameter-based Chinchilla Scaling

**Description**: Extract parameter count from model names, apply era-aware token estimation.

**Token Estimation Heuristics**:

| Model Era | Parameter Range | Tokens/Parameter | Rationale |
|---|---|---|---|
| Modern (2024+) | ≥100B | 15 | Conservative for large models |
| Modern (2024+) | <100B | 20 | Standard Chinchilla ratio |
| Mid-era (2022-2023) | All | 12 | Pre-Llama 3 training practices |
| Specialized (code/instruct) | All | 18 | Extended training typical |
| Chinese models | All | 16 | Well-trained regional models |

**Example**:
```
Model: qwen2.5_72b
Parameters: 72,000,000,000 (extracted)
Era: Modern (2024)
Tokens: 72B × 20 = 1.44T
FLOP = 6 × 7.2e10 × 1.44e12 = 6.22e23
Confidence: MEDIUM
```

#### 3.2.4 Method 3: Multi-Benchmark Interpolation

**Description**: Estimate FLOP from benchmark scores using power law scaling relationships calibrated against reference models.

**Scaling Formula**:
```
FLOP_model = FLOP_reference × (Score_model / Score_reference)^α
```

Currently α = 3.0 for all benchmark types based on implementation. This value was chosen empirically but lacks theoretical justification.

**Reference Models**:

| Benchmark | Reference Model | Reference Score | Reference FLOP | Source |
|---|---|---|---|---|
| lmarena_score | llama_3.1_405b | 1335 | 3.8e25 | Meta disclosure |
| openlm_arena_elo | llama_3.1_405b | 1286 | 3.8e25 | Meta disclosure |
| mmlu_pro_score | llama_3.1_405b | 73.2% | 3.8e25 | Meta disclosure |

**Aggregation Process**:
1. Calculate individual estimates for each available benchmark
2. Weight by confidence (HIGH=1.0, MEDIUM=0.7, LOW=0.4)
3. Check agreement (estimates within 2× boost confidence)
4. Return weighted average with aggregated confidence

### 3.3 Confidence Scoring

Confidence levels reflect both data quality and estimation method:

| Level | Criteria | Typical Sources |
|---|---|---|
| **HIGH** | Published specifications, official disclosures, Epoch AI high-confidence | Company blogs, papers |
| **MEDIUM** | Reliable estimation methods, multiple agreeing benchmarks | Parameter extraction, benchmark consensus |
| **LOW** | Single benchmark, distant interpolation, older models | Limited data |
| **SPECULATIVE** | Heuristics only, major assumptions | Name parsing without context |

### 3.4 Audit Trail

Every estimation maintains complete provenance:
- Primary method and calculation details in `reasoning`
- Alternative attempts in `alternative_methods`  
- Data sources in `sources`
- Manual adjustments in `notes`

## 4. System Architecture

For complete technical details, repository structure, and implementation specifics, see the project [README.md](../README.md).

## 5. Key Considerations for Researchers

**Important Limitations**:
- FLOP estimates have inherent uncertainty (factor of 2-3× typical for MEDIUM confidence)
- Confidence levels indicate reliability of estimates  
- Coverage limited to publicly disclosed or inferrable models
- Training compute only (excludes inference, fine-tuning)

**Dataset Properties**:
- Manual verification required for all included models
- Dataset updated as new frontier models are identified
- Complete audit trail maintained for all estimates