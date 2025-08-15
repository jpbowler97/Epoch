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

### Core Identification Fields

| Field | Type | Definition | Example | Population Method |
|---|---|---|---|---|
| **model** | string | Normalized model name in snake_case format | `gpt_4o`, `claude_3_opus` | Automatic (scraping + normalization) |
| **developer** | string | Organization that developed the model | `OpenAI`, `Anthropic` | Automatic (source data) |
| **release_date** | date | First public announcement or API availability | `2024-05-13` | Automatic/Manual |

### Model Specifications

| Field | Type | Definition | Example | Population Method |
|---|---|---|---|---|
| **parameters** | float | Total number of model parameters | `1.76e+12` (1.76T) | Automatic (when disclosed) |
| **parameter_source** | string | Source and method for parameter count | `known_specification:gpt_4` | Automatic |

### FLOP Estimation Fields

| Field | Type | Definition | Example | Population Method |
|---|---|---|---|---|
| **training_flop** | float | Estimated training compute in FLOP | `2.1e+25` | Automatic (estimation pipeline) |
| **confidence** | enum | Confidence level of FLOP estimate | `high`, `medium`, `low`, `speculative` | Automatic |
| **confidence_explanation** | string | Detailed reasoning for confidence assignment | `Official disclosure from Meta` | Automatic |
| **estimation_method** | enum | Primary method used for FLOP calculation | `epoch_estimate`, `scaling_laws`, `benchmark_based` | Automatic |
| **alternative_methods** | string | Other estimation attempts and results | `Scaling Laws: 8.4e+24 (Medium)` | Automatic |

### Classification Fields

| Field | Type | Definition | Example | Population Method |
|---|---|---|---|---|
| **threshold_classification** | enum | Relationship to 10²⁵ FLOP threshold | `high_confidence_above_1e25` | Automatic |
| **status** | enum | Overall inclusion status | `confirmed_above_1e25`, `uncertain` | Automatic/Manual |

### Documentation Fields

| Field | Type | Definition | Example | Population Method |
|---|---|---|---|---|
| **reasoning** | text | Detailed calculation or estimation logic | `Chinchilla scaling: 6 × 405B params × 15T tokens` | Automatic |
| **sources** | text | URLs and references for data | `https://arxiv.org/abs/2407.21783` | Automatic |
| **notes** | text | Manual annotations and verification reasoning | `Verified via official Meta blog post` | Manual |

### Metadata Fields

| Field | Type | Definition | Example | Population Method |
|---|---|---|---|---|
| **verified** | boolean | Manual verification flag (y/n) | `y` | Manual (review process) |
| **last_updated** | timestamp | Last modification timestamp | `2025-02-01T12:00:00Z` | Automatic |
| **blacklist_status** | string | Developer transparency flag | `capped_insufficient_transparency` | Automatic |
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

This relationship holds across model scales from 70M to 500B+ parameters, validated by empirical results from DeepMind, OpenAI, and Anthropic.

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

Where α (scaling exponent) varies by benchmark type:
- ELO-based scores: α = 3.0
- Percentage scores (MMLU): α = 2.5  
- Specialized benchmarks: α = 2.0

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

### 3.4 Validation and Quality Control

#### 3.4.1 Range Validation

FLOP estimates must fall within physically plausible bounds:
- Minimum: 1e20 FLOP (small research models)
- Maximum: 1e28 FLOP (current infrastructure limits)
- Estimates outside range trigger manual review

#### 3.4.2 Consistency Checks

- Models from same family should show monotonic scaling
- Contemporary models should cluster in FLOP ranges
- Benchmark scores should correlate with FLOP estimates

#### 3.4.3 Audit Trail

Every estimation maintains complete provenance:
- Primary method and calculation details in `reasoning`
- Alternative attempts in `alternative_methods`
- Data sources in `sources`
- Manual adjustments in `notes`

### 3.5 Special Cases

#### 3.5.1 Mixture of Experts (MoE)

MoE models report both total and active parameters. FLOP calculations use:
- Training: Total parameters (all experts trained)
- Inference: Active parameters (subset activated)

Example: Mixtral 8×7B = 47B total parameters for training FLOP

#### 3.5.2 Continued Pre-training

Models with multiple training phases aggregate compute:
```
Total FLOP = Initial training + Continued training + Fine-tuning
```

#### 3.5.3 Multimodal Models

Vision and multimodal models include:
- Text token processing
- Image patch encoding  
- Cross-modal attention

Typically 1.2-1.5× compute vs text-only equivalent

## 4. System Architecture

### 4.1 Repository Structure

The complete system available at: https://github.com/epochai/model-tracker

```
src/epoch_tracker/
├── models/          # Pydantic schemas for data validation
├── scrapers/        # Modular scraper implementations
├── estimation/      # FLOP calculation methods
├── storage/         # JSON/CSV persistence layer
└── query/           # Data filtering and analysis

scripts/
├── run.py                      # Main CLI interface
├── data_collection/            # Stage 1 scripts
├── data_processing/            # Stage 2-3 scripts
└── curation/                   # Stage 4 scripts

configs/
├── model_name_mapping.yaml     # Staging→Published mappings
├── benchmark_references.json   # Reference models for interpolation
└── developer_blacklist.json    # Transparency-based capping

data/
├── scraped/                    # Raw collected data
├── estimated/                  # Models with FLOP estimates
├── clean/                      # Candidate models for review
└── staging/                    # Verified models for production
```

### 4.2 Extensibility

The system designed for easy extension:

**Adding New Scrapers**:
1. Inherit from `BaseScraper` class
2. Implement `scrape_models()` method
3. Add configuration to `configs/scrapers/`

**Adding New Benchmarks**:
1. Edit `configs/benchmark_references.json`
2. Include reference model and scaling parameters
3. No code changes required

**Adding New Estimation Methods**:
1. Implement in `ComputeEstimator` class
2. Add to hierarchy in `estimate_flops.py`
3. Define confidence mapping

## 5. Usage Guidelines

### 5.1 For Researchers

This dataset enables research on:
- AI scaling trends and compute requirements
- Efficiency improvements over time
- Regional differences in AI development
- Correlation between compute and capabilities

**Important Considerations**:
- FLOP estimates have inherent uncertainty (factor of 2-3× typical)
- Confidence levels indicate reliability
- Manual verification ongoing for borderline cases
- Dataset updates monthly with new models

### 5.2 For Contributors

To contribute improved estimates or new models:
1. Review existing estimation methods
2. Check `configs/model_name_mapping.yaml` for naming conventions
3. Submit pull requests with:
   - Source documentation
   - Calculation methodology
   - Confidence justification

### 5.3 Citation

When using this dataset in research:

```bibtex
@dataset{epoch_ai_models_1e25_2025,
  title={AI Models Above 1e25 FLOP Dataset},
  author={Epoch AI},
  year={2025},
  url={https://github.com/epochai/model-tracker},
  note={Semi-automated tracking of frontier AI systems}
}
```

## 6. Limitations and Future Work

### 6.1 Current Limitations

- **Coverage**: Limited to publicly disclosed or inferrable models
- **Accuracy**: FLOP estimates uncertain for closed models
- **Latency**: 1-2 week delay for new model inclusion
- **Scope**: Training compute only (excludes inference, fine-tuning)

### 6.2 Planned Improvements

- **Real-time tracking**: API integration for automatic updates
- **Inference compute**: Add deployment cost estimates
- **Hardware tracking**: Detailed GPU/TPU configuration data
- **Carbon footprint**: Energy consumption and emissions

## 7. Appendices

### 7.1 Frequently Asked Questions

**Q: Why 10²⁵ FLOP as the threshold?**
A: This represents approximately $1-5 million in compute costs (2024 prices) and typically indicates frontier model development efforts.

**Q: How accurate are the FLOP estimates?**
A: HIGH confidence estimates typically within 50% of true value. MEDIUM confidence within factor of 2-3×. LOW confidence order-of-magnitude estimates.

**Q: What about proprietary models?**
A: We include all models with sufficient public information for estimation. Completely proprietary models without any disclosure excluded.

### 7.2 Glossary

- **FLOP**: Floating-point operation, the basic unit of computational work
- **Chinchilla scaling**: Optimal compute allocation discovered by DeepMind
- **ELO score**: Relative performance rating from competitive evaluation
- **Staging dataset**: Manually verified models ready for production use
- **Confidence level**: Reliability indicator for FLOP estimates

### 7.3 Contact Information

For questions, corrections, or contributions:
- GitHub Issues: https://github.com/epochai/model-tracker/issues
- Email: research@epoch.ai
- Dataset Updates: https://epoch.ai/data/models

---

*This documentation reflects the methodology as of February 2025. For the latest updates, consult the GitHub repository.*