# Epoch AI Model Tracker

Semi-automated system for tracking AI models trained with over 1e25 FLOP.

## Overview

This project builds a structured system to improve Epoch's tracking of large-scale AI models. It includes:

- **Manual + automated data collection** from LMArena leaderboards and other sources
- **FLOP estimation engine** using scaling laws, benchmark interpolation, and model specifications
- **Structured data schemas** for consistent model metadata and confidence scoring
- **Core dataset management** for curating models above the 1e25 FLOP threshold with manual verification

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data from sources (uses manual CSV files + web scraping when available)
python scripts/get_latest_model_data.py

# 3. Apply FLOP estimations to scraped models
python scripts/estimate_flops.py --update

# 4. Generate/refresh core dataset of models above 1e25 FLOP threshold
python scripts/refresh_core_dataset.py

# 5. Review new candidate models (semi-automatic curation)
python scripts/review_candidates.py
```

### Model Review Workflow

After running the automated pipeline, use the semi-automatic review process to curate the final dataset:

```bash
python scripts/review_candidates.py
```

For each new candidate model, you can:
- **`y`** - Add to staging dataset (confirming it's above 1e25 FLOP)
- **`n`** - Move to below_1e25_flop.csv with verified=y (confirming it's below threshold)
- **`skip`** - Skip for now (can review later)

**Key feature**: Models marked with `verified=y` are permanently protected from automated pipeline changes, ensuring your manual decisions are preserved across future runs.

## Data Pipeline Workflow

The system follows a **four-stage data processing pipeline**:

### Stage 1: Raw Data Collection
- **Manual data**: Place manually collected CSV files in `data/raw/lmarena/text/`
- **Automated scraping**: `python scripts/fetch_lmarena_data.py` (when sources are accessible)
- **Data refresh**: `python scripts/get_latest_model_data.py` processes all available data sources
- **Output**: Raw model data in `data/scraped/`

### Stage 2: FLOP Estimation
- **Apply estimations**: `python scripts/estimate_flops.py --update`
- **Methods used**: Known model specs, benchmark score interpolation, parameter size heuristics
- **Confidence scoring**: High/Medium/Low/Speculative based on estimation method and data quality
- **Output**: Consolidated models with FLOP estimates in `data/estimated/`

### Stage 3: Core Dataset Curation
- **Generate dataset**: `python scripts/refresh_core_dataset.py` creates `data/clean/above_1e25_flop.csv`
- **Candidate identification**: Models above threshold automatically flagged for review
- **Output**: Candidate models in `data/clean/` ready for manual verification

### Stage 4: Manual Review & Staging
- **Interactive review**: `python scripts/review_candidates.py` compares candidates against staging dataset
- **Manual decisions**: For each candidate, decide to add to staging, move to below threshold, or skip
- **Verification protection**: Models marked `verified=y` are protected from automated pipeline changes
- **Output**: Curated staging dataset in `data/staging/above_1e25_flop_staging.csv` ready for production

**📚 Documentation:**
- **[Developer Guide](CLAUDE.md)** - Development setup, architecture, and current system status
- **[Core Dataset Management](docs/core_dataset_management.md)** - Detailed guide to curating models above 1e25 FLOP
- **[Review Workflow](docs/review_workflow.md)** - Semi-automatic candidate review process and manual curation
- **[Data Pipeline](docs/data_pipeline.md)** - Complete data flow from collection to curation
- **[FLOP Estimation Guide](docs/flop_estimation_guide.md)** - FLOP calculation methods and implementation details
- **[Assignment](Assignment.md)** - Project objectives and success criteria

## Architecture

```
src/epoch_tracker/
├── models/          # Data schemas (Model, ModelCollection)
├── scrapers/        # Data collection (LMArena, Papers with Code)
├── query/           # Query engine and output formatters
├── estimation/      # FLOP calculation algorithms (scaling laws, benchmarks)
├── storage/         # Data persistence (JSON, CSV export)
├── config/          # Configuration management
└── utils/           # HTTP client, date parsing, etc.

scripts/
├── get_latest_model_data.py  # Collect data from all configured sources
├── fetch_lmarena_data.py     # Automated LMArena web scraping
├── estimate_flops.py         # Apply FLOP estimations to models
├── refresh_core_dataset.py   # Generate/refresh curated dataset above 1e25 FLOP
├── review_candidates.py      # Semi-automatic review process for manual curation
├── query_models.py           # Interactive data exploration (optional)
└── manual_model_entry.py     # Manual model entry interface

data/
├── scraped/        # Raw scraped model data (Stage 1)
├── estimated/      # Models with FLOP estimates (Stage 2)
├── clean/          # Curated datasets for manual verification (Stage 3)
├── staging/        # Manually reviewed models ready for production (Stage 4)
└── query_results/  # CSV exports and query outputs
```

## Current Data Sources

- **LMArena** (manual + automated): Leaderboard scores and model rankings
- **OpenLM Arena** (automated): Chatbot arena leaderboard data
- **Manual datasets**: CSV files in `data/raw/*/` directories
- **Epoch AI tracker**: Manual overrides for frontier models with authoritative FLOP values

## FLOP Estimation Methods

1. **Epoch Data**: Use data collected by Epoch [here](https://epoch.ai/data-insights/models-over-1e25-flop)
2. **Benchmark Score Interpolation**: ELO rating → FLOP estimation using reference models
3. **Parameter Size Heuristics**: Extract model size from names and apply scaling laws
4. **Confidence Levels**: High (published specs) → Medium (reliable estimates) → Low (interpolated) → Speculative (heuristics)

## For more info....

For detailed usage instructions, see the **[Usage Guide](docs/usage.md)**.
For development and architecture details, see **[CLAUDE.md](CLAUDE.md)**.
