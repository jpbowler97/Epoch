# Documentation Changes for Epoch AI Models Database

## Overview

This document specifies changes needed to update the existing [Epoch AI Models Documentation](https://epoch.ai/data/ai-models-documentation) to incorporate the new semi-automated methodology for tracking models above 1e25 FLOP.


## Inclusion Section Changes

**Add new subsection after existing notability criteria:**

### Computational Threshold Models (≥1e25 FLOP)

For frontier models, we use a **semi-automated inclusion process**:
- Automated collection from leaderboards (LMArena, OpenLM Arena) and benchmark sites
- Browser-interactive data collection using AI agents for complex sites
- Threshold-based filtering: models ≥1e25 FLOP flagged for manual review
- Expert verification required for all candidates before inclusion
- Manual decisions protected from automated overwrites via `verified=y` flag

This ensures comprehensive coverage of frontier AI systems while maintaining quality through human verification.

## Database Updates Section Changes

This section would largely be replaced by the documentation [here](../docs/ai_models_above_1e25_documentation.md) giving a more detailed overview of how the database is updated and maintained.

## Estimation Section Changes

**Incorporate into "Estimating compute" subsection:**

**Hierarchical method priority:**

1. **Authoritative Manual Estimates (HIGH confidence)** - Expert analysis of official disclosures
2. **Chinchilla Scaling (MEDIUM confidence)** - `6 × parameters × tokens` when parameter count is known
3. **Benchmark Interpolation (LOW confidence)** - Benchmark score comparison against model(s) with known FLOPs
   - *Caveat: Increasingly unreliable due to efficiency improvements over time*

**Add developer exclusion criteria:**
Some developers capped at 9.9×10²⁴ FLOP based on limited computational resources or lack of demonstrated 1e25+ FLOP training capability.

## Records Section Changes

The following changes would be made to the existing [AI models documentation table](https://epoch.ai/data/ai-models-documentation#records):

### Existing Fields with Enhanced Definitions

| Column | Type | Definition | Example | Coverage |
|--------|------|------------|---------------------------|----------|
| Confidence | Enum | **Refined definition:** Structured levels (HIGH/MEDIUM/LOW/SPECULATIVE) with strict criteria based on estimation method hierarchy - see [detailed definitions](../docs/ai_models_above_1e25_documentation.md) | `HIGH` | Existing |
| Training compute estimation method | Text | **Refined definition:** Currently this is a category e.g. "Hardware", we would now include the detailed calculation behind the estimate | "Known model specification 'deepseek_v3': Chinchilla scaling law: 6 × 671,000,000,000 params × 14,800,000,000,000 tokens = 5.96e+25 FLOP" | Existing |

### New Fields to Add

| Column | Type | Definition | Example from Llama 2-70B | Coverage |
|--------|------|------------|---------------------------|----------|
| Confidence explanation | Text | Detailed reasoning for confidence level assignment beyond the calculation | `Known parameters with documented training tokens` | TBD |
| Alternative methods | Text | Results from other estimation attempts with their confidence levels | `Benchmark Based: 1.46e+25 (Low)` | TBD |
| Verified | Boolean | Manual expert verification completed to protect against automated overwrites | `false` (not manually verified) | TBD |
| Threshold classification | Enum | Relationship to 1e25 FLOP threshold (high_confidence_above_1e25, likely_above_1e25, uncertain, likely_below_1e25, confirmed_below_1e25) | `confirmed_below_1e25` | TBD |
| Status | Enum | Overall inclusion determination (confirmed_above, likely_above, uncertain, confirmed_below, likely_below) | `confirmed_below_1e25` | TBD |
| Collection method | Enum | How data was obtained (automated_scraping, browser_interactive, manual_research, company_disclosure) | `automated_scraping` | TBD |
| Parameter source | Text | Origin of parameter count (known_specification, extracted_from_name, company_disclosure) | `known_specification:llama_2_70b` | TBD |
| Exclusion reason | Text | Why developer estimates were resource-capped (for transparency) | `null` (Meta not resource-capped) | TBD |
| Last updated | Timestamp | Most recent data refresh timestamp | `2025-08-15T18:13:34.256770` | TBD |

*Coverage percentages marked as "TBD" will be updated after further testing and implementation of the enhanced data collection pipeline.*


## Generic Changes

**Throughout documentation, emphasize:**
- Semi-automated approach combining automation with human expertise
- Complete audit trail from data collection to final inclusion

There are some other existing fields which could be augmented based on the semi-automated process, but the above are the key changes.
