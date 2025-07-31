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
```

## Data Pipeline Workflow

The system follows a **three-stage data processing pipeline**:

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
- **Manual verification**: Mark models as `verified=y` after human review
- **Incremental updates**: Refresh preserves manually verified entries while updating estimates

**📚 Documentation:**
- **[Developer Guide](CLAUDE.md)** - Development setup, architecture, and current system status
- **[Core Dataset Management](docs/core_dataset_management.md)** - Detailed guide to curating models above 1e25 FLOP
- **[Data Pipeline](docs/data_pipeline.md)** - Complete data flow from collection to curation
- **[FLOP Estimation Guide](docs/flop_estimation_guide.md)** - FLOP calculation methods and implementation details
- **[Assignment](Assignment.md)** - Project objectives and success criteria
- **[Implementation Plan](Plan.md)** - Development roadmap to production

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
└── query_models.py           # Interactive data exploration (optional)

data/
├── scraped/        # Raw scraped model data (Stage 1)
├── estimated/      # Models with FLOP estimates (Stage 2)
├── clean/          # Curated datasets for manual verification (Stage 3)
└── query_results/  # CSV exports and query outputs
```

## Current Data Sources

- **LMArena** (manual + automated): Leaderboard scores and model rankings
- **Papers with Code**: Academic model releases and benchmarks
- **Manual datasets**: CSV files in `data/raw/*/` directories

## FLOP Estimation Methods

1. **Known Model Specifications**: Direct calculation using published parameters and training tokens
2. **Benchmark Score Interpolation**: ELO rating → FLOP estimation using reference models
3. **Parameter Size Heuristics**: Extract model size from names and apply scaling laws
4. **Confidence Levels**: High (published specs) → Medium (reliable estimates) → Low (interpolated) → Speculative (heuristics)

## Next Steps

For detailed usage instructions, see the **[Usage Guide](docs/usage.md)**.
For development and architecture details, see **[CLAUDE.md](CLAUDE.md)**.
