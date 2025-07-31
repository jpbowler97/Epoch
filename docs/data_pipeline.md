# Data Pipeline Architecture

This document explains the complete data flow from raw model information to curated datasets, including the technical implementation, design decisions, and operational considerations for each stage.

## Pipeline Overview

The Epoch AI Model Tracker implements a **three-stage data processing pipeline** designed to transform raw model information into high-quality, manually-verified datasets:

```
[Stage 1: Collection] → [Stage 2: Estimation] → [Stage 3: Curation]
     Raw Data              FLOP Estimates         Verified Dataset
```

### Design Principles

1. **Separation of Concerns**: Each stage has a distinct responsibility and can be run independently
2. **Idempotent Operations**: Scripts can be re-run safely without data corruption
3. **Audit Trail**: Complete lineage from raw data to final output
4. **Human-in-the-Loop**: Combines automation with manual verification where needed
5. **Configurable Sources**: Easy to add new data sources without code changes

## Stage 1: Raw Data Collection

### Purpose
Collect model information from various sources and normalize it into a consistent schema.

### Input Sources
- **Manual CSV files** in `data/raw/lmarena/text/` for example
- **Automated web scraping** from LMArena

### Technical Implementation

#### Primary Script: `get_latest_model_data.py`

#### Scraper Architecture
All scrapers inherit from `BaseScraper` and implement:
- `scrape_models()`: Collect raw data
- `normalize_data()`: Convert to `Model` objects
- `get_source_name()`: Identify data source

#### Data Flow
```
Raw Sources
├── data/raw/lmarena/text/*.csv (manual)
├── https://lmarena.ai/ (automated)
├── https://paperswithcode.com/ (automated)
└── Manual entry scripts

    ↓ [Scraping & Normalization]

data/scraped/
├── lmarena_models.json
├── openlm_arena_models.json
├── paperswithcode_models.json
└── manual_models.json
```

### Output Schema
Each scraped file contains a `ModelCollection` with:
```json
{
  "metadata": {
    "saved_at": "2025-01-31T12:00:00Z",
    "source": "lmarena",
    "last_updated": "2025-01-31T12:00:00Z",
    "model_count": 150
  },
  "models": [
    {
      "name": "gpt-4",
      "developer": "OpenAI",
      "benchmarks": {"lmarena_score": 1200},
      "sources": ["https://lmarena.ai/"],
      "last_updated": "2025-01-31T12:00:00Z"
    }
  ]
}
```


## Stage 2: FLOP Estimation

### Purpose
Apply various estimation methods to calculate training FLOP for each model and assign confidence levels.

### Technical Implementation

#### Primary Script: `estimate_flops.py`
```python
def main():
    # Load all scraped models
    all_models = storage.load_all_scraped_models()
    
    estimator = ComputeEstimator()
    
    for model in all_models:
        if should_update_estimate(model):
            estimate = estimate_model_flops(model, estimator)
            if estimate:
                model.update_flop_estimate(
                    flop=estimate['flop'],
                    confidence=estimate['confidence'],
                    method=estimate['method'],
                    reasoning=estimate['reasoning']
                )
    
    # Apply threshold classification
    classify_by_threshold(all_models)
    
    # Save consolidated results
    collection = ModelCollection(models=all_models, source="consolidated_estimated")
    storage.save_models(collection, "estimated_models", stage="estimated")
```

#### Estimation Methods (Priority Order)

1. **Known Model Specifications**
   ```python
   KNOWN_MODEL_SPECS = {
       "llama-3.1-405b": (405B_params, 15T_tokens, ConfidenceLevel.HIGH),
       "gpt-4": (1.76T_params, 13T_tokens, ConfidenceLevel.MEDIUM)
   }
   ```

2. **Benchmark Score Interpolation**
   ```python
   def estimate_from_benchmark_score(model, estimator):
       elo_score = model.benchmarks.get('lmarena_score')
       reference_models = BENCHMARK_REFERENCES['lmarena_score']
       return estimator.estimate_from_benchmark_elo(elo_score, reference_models)
   ```

3. **Parameter Size Heuristics**
   ```python
   def extract_model_size(model_name):
       # Extract "405B", "70B" etc. from model names
       patterns = [r'(\d+\.?\d*)b(?:illion)?', r'(\d+\.?\d*)t(?:rillion)?']
       # Apply 20x tokens per parameter ratio
   ```

#### Confidence Assignment

| Method | Confidence | Criteria |
|--------|------------|----------|
| Known specs | High | Published training details |
| Benchmark + specs | Medium | ELO + parameter count |
| Benchmark only | Low | ELO interpolation |
| Name heuristics | Speculative | Extracted from model name |

#### Threshold Classification

Models are classified based on FLOP estimates:

```python
HIGH_CONFIDENCE_ABOVE_THRESHOLD = 5e25  # >= 5x threshold
HIGH_CONFIDENCE_BELOW_THRESHOLD = 5e24  # <= 0.5x threshold

def classify_by_threshold(flop_estimate):
    if flop_estimate >= HIGH_CONFIDENCE_ABOVE_THRESHOLD:
        return ThresholdClassification.HIGH_CONFIDENCE_ABOVE
    elif flop_estimate <= HIGH_CONFIDENCE_BELOW_THRESHOLD:
        return ThresholdClassification.HIGH_CONFIDENCE_BELOW
    else:
        return ThresholdClassification.NOT_SURE  # Needs verification
```

### Data Flow
```
data/scraped/*.json
    ↓ [Deduplication by (name, developer)]
    ↓ [FLOP Estimation: Known Specs → Benchmarks → Heuristics]
    ↓ [Confidence Assignment: High → Medium → Low → Speculative]
    ↓ [Threshold Classification: Above/Below/Uncertain]
    ↓
data/estimated/estimated_models.json (418 models total)
```

### Quality Assurance
- **Deduplication**: Prefer higher confidence estimates for same model
- **Range validation**: FLOP estimates must be within reasonable bounds
- **Method tracking**: Complete audit trail of estimation approach
- **Confidence scoring**: Systematic confidence assignment

## Stage 3: Core Dataset Curation

### Purpose
Create a curated dataset of models above 1e25 FLOP that combines automated estimation with human verification.

### Technical Implementation

#### Primary Script: `refresh_core_dataset.py`
```python
def main():
    # Load estimated models
    collection = storage.load_models("estimated_models", stage="estimated")
    
    # Filter for candidates (HIGH_CONFIDENCE_ABOVE + NOT_SURE)
    candidates = get_candidate_models(collection.models)
    
    if csv_exists():
        # Refresh mode: preserve verified entries
        df = refresh_existing_csv(existing_df, candidates, csv_path)
    else:
        # Create mode: new dataset from scratch
        df = create_new_csv(candidates, csv_path)
```

#### Candidate Selection Logic
```python
def get_candidate_models(models):
    return [
        model for model in models 
        if model.threshold_classification in [
            ThresholdClassification.HIGH_CONFIDENCE_ABOVE,  # Definitely above
            ThresholdClassification.NOT_SURE                # Needs verification
        ]
    ]
    # Excludes HIGH_CONFIDENCE_BELOW (definitely below threshold)
```

#### Smart Refresh Algorithm
```python
def refresh_existing_csv(existing_df, new_models, csv_path):
    updated_rows = []
    
    for _, row in existing_df.iterrows():
        if row.get('verified') == 'y':
            # Preserve manually verified entries unchanged
            updated_rows.append(row.to_dict())
            verified_count += 1
        else:
            # Update with latest estimates if available
            updated_row = find_updated_model(row, new_models)
            updated_rows.append(updated_row)
            updated_count += 1
    
    # Add new candidates not in existing CSV
    add_new_candidates(new_models, existing_df, updated_rows)
    
    return create_dataframe(updated_rows)
```

### Data Flow
```
data/estimated/estimated_models.json (418 models)
    ↓ [Filter: HIGH_CONFIDENCE_ABOVE + NOT_SURE]
    ↓ [400 candidate models]
    ↓
data/clean/above_1e25_flop.csv
    ↓ [Manual Verification: verified=y]
    ↓ [15 verified models]
    ↓
Epoch Production Dataset
```

### Verification Workflow

#### Automated Processing
1. **Candidate identification**: Models likely above 1e25 FLOP
2. **CSV generation**: Structured format for manual review
3. **Incremental updates**: Preserve manual work during refresh

#### Manual Verification
1. **Research phase**: Check papers, blog posts, announcements
2. **Verification**: Mark `verified=y` for confirmed models
3. **Documentation**: Add notes explaining verification reasoning
4. **Quality control**: Cross-reference multiple sources

### Output Schema
The final CSV includes comprehensive model information:
- **Metadata**: name, developer, release_date, parameters
- **FLOP estimates**: training_flop, confidence, estimation_method
- **Classification**: threshold_classification, status
- **Provenance**: reasoning, sources, last_updated
- **Verification**: verified, notes (manual fields)

## Pipeline Operations

### Running the Complete Pipeline

```bash
# Full pipeline execution
python scripts/get_latest_model_data.py     # Stage 1: Collection
python scripts/estimate_flops.py --update  # Stage 2: Estimation  
python scripts/refresh_core_dataset.py     # Stage 3: Curation
```

### Incremental Updates

```bash
# Daily: Update raw data
python scripts/get_latest_model_data.py

# Weekly: Refresh estimates
python scripts/estimate_flops.py --update

# As needed: Refresh core dataset (preserves manual work)
python scripts/refresh_core_dataset.py
```

### Monitoring and Validation

Each stage provides detailed statistics:
```bash
# Stage 1: Collection statistics
Scraped 150 models from LMArena
Scraped 200 models from OpenLM Arena
Total unique models: 280

# Stage 2: Estimation statistics  
Updated 45 models with FLOP estimates
High confidence above 1e25: 8 models
Medium confidence: 32 models
Speculative estimates: 178 models

# Stage 3: Curation statistics
Total candidate models: 400
Manually verified: 15
High confidence above 1e25: 8
Uncertain (need verification): 377
```

## Directory Structure

### Complete Data Layout
```
data/
├── raw/                    # Original source data
│   └── lmarena/text/      # Manual CSV files
├── scraped/               # Stage 1 output
│   ├── lmarena_models.json
│   ├── openlm_arena_models.json
│   └── manual_models.json
├── estimated/             # Stage 2 output
│   └── estimated_models.json
├── clean/                 # Stage 3 output
│   └── above_1e25_flop.csv
└── query_results/         # Analysis outputs
    └── models_above_1e25_flop_*.csv
```

### File Lifecycle
- **Raw files**: Never modified automatically
- **Scraped files**: Overwritten on each collection run
- **Estimated files**: Consolidated from all scraped data
- **Clean files**: Incrementally updated, preserving manual work
- **Query results**: Generated on-demand, timestamped

## Error Handling and Recovery

### Stage Isolation
Each stage can recover independently:
- **Stage 1 failure**: Re-run collection without affecting estimates
- **Stage 2 failure**: Estimates can be recalculated from scraped data
- **Stage 3 failure**: Manual verification work is preserved

### Data Validation
- **Schema validation**: All data conforms to Pydantic models
- **Range checks**: FLOP values within reasonable bounds
- **Consistency checks**: Cross-validation between stages
- **Audit logging**: Complete tracking of all operations

### Backup and Recovery
- **Version control**: All code and configuration tracked
- **Data snapshots**: Key datasets committed to git
- **Manual work preservation**: Verified entries never lost
- **Rollback capability**: Can revert to previous states

## Performance Considerations

### Scalability
- **Incremental processing**: Only update changed data
- **Parallel execution**: Scrapers can run concurrently
- **Memory efficiency**: Stream processing for large datasets
- **Caching**: Avoid redundant API calls

### Optimization Opportunities
- **Database backend**: Replace JSON files for better performance
- **Distributed processing**: Scale across multiple machines
- **Real-time updates**: Stream processing for continuous updates
- **ML acceleration**: GPU-based similarity calculations

This comprehensive pipeline ensures reliable, auditable processing of AI model data from diverse sources to high-quality curated datasets suitable for research and production use.