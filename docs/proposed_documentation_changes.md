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
4. **Hardware-based (SPECULATIVE)** - GPU specifications and training duration

**Add developer exclusion criteria:**
Some developers capped at 9.9×10²⁴ FLOP based on limited computational resources or lack of demonstrated 1e25+ FLOP training capability.

## Records Section Changes

**Add these new fields to the schema table:**

### FLOP Estimation Fields
- `training_flop` (float): Estimated training compute in FLOP
- `confidence` (enum): HIGH/MEDIUM/LOW/SPECULATIVE reliability level - see [here](../docs/ai_models_above_1e25_documentation.md) for detailed definitions
- `confidence_explanation` (text): Detailed reasoning for confidence assignment
- `estimation_method` (enum): Primary method used (epoch_estimate, scaling_laws, benchmark_based)
- `alternative_methods` (text): Results from other estimation attempts
- `reasoning` (text): Detailed calculation explanation

### Verification Fields  
- `verified` (boolean): Manual expert verification completed
- `threshold_classification` (enum): Relationship to 1e25 FLOP threshold
- `status` (enum): Overall inclusion determination
- `sources` (text): All data source URLs
- `collection_method` (enum): How data was obtained
- `last_updated` (timestamp): Most recent data refresh

### Quality Assurance Fields
- `exclusion_reason` (text): Why developer estimates were resource-capped
- `notes` (text): Manual annotations and verification reasoning

## Generic Changes

**Throughout documentation, emphasize:**
- Semi-automated approach combining automation with human expertise
- Complete audit trail from data collection to final inclusion

These changes document the methodology actively used since [INSERT DATE] for tracking the most computationally intensive AI systems.
