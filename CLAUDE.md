# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a system for tracking AI models trained with over 1e25 FLOP. The project aims to build a structured, semi-automated system that includes:

- A clean schema for model data stored in JSON format
- Scrapers for public leaderboards and model releases from frontier AI companies
- Compute estimation logic based on scaling laws and benchmark comparisons
- Core dataset management for curating models above 1e25 FLOP with manual verification
- Semi-automated pipelines for ingesting and flagging model candidates

## Common Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (includes pytest, black, isort, flake8)
pip install -e ".[dev]"

# Run tests
pytest

# Run all tests with coverage
pytest --cov=src/epoch_tracker

# Code formatting
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Linting
flake8 src/ scripts/ tests/

# Primary system entry points (using main script)
python scripts/run.py collect-all              # Refresh all model data
python scripts/run.py collect-all --update-claude-sites  # Refresh with Claude updates
python scripts/run.py update-claude            # Update only Claude-managed sites
python scripts/run.py refresh-dataset          # Generate/refresh curated dataset above 1e25 FLOP
python scripts/run.py query --above-threshold --format table  # Query results
python scripts/run.py manual-entry             # Manual model entry interface
python scripts/run.py validate                 # Validate against Epoch's data

# Claude-specific commands for JavaScript-heavy sites
python scripts/run.py update-claude            # Update Claude-managed sites (shows simple command)
python scripts/run.py update-claude --sites superclue --force  # Force update specific sites

# Individual pipeline stages
python scripts/run.py collect-all              # Stage 1: Data collection
python scripts/run.py estimate-flops --update  # Stage 2: FLOP estimation
python scripts/run.py refresh-dataset          # Stage 3: Core dataset curation
python scripts/run.py review-candidates        # Stage 4: Manual review & staging

# Alternative: Direct script access (legacy support)
python scripts/data_collection/get_latest_model_data.py         # Stage 1: Data collection
python scripts/data_processing/estimate_flops.py --update       # Stage 2: FLOP estimation
python scripts/data_processing/refresh_core_dataset.py          # Stage 3: Core dataset curation
python scripts/curation/review_candidates.py                    # Stage 4: Manual review & staging
```

## Architecture

The system follows a modular, extensible design:

### Core Components
- **Schema** (`src/epoch_tracker/models/`): Pydantic models for structured data validation
- **Scrapers** (`src/epoch_tracker/scrapers/`): Pluggable data collection from multiple sources
- **Query Engine** (`src/epoch_tracker/query/`): Filtering, sorting, and output formatting
- **Estimation Logic** (`src/epoch_tracker/estimation/`): Multiple FLOP computation methods
- **Storage** (`src/epoch_tracker/storage/`): JSON persistence with deduplication
- **Configuration** (`configs/`): YAML-based settings management

### Key Architectural Patterns
- **Extensible scrapers**: New data sources inherit from `BaseScraper`
- **Pluggable estimators**: FLOP estimation supports multiple methodologies 
- **Type-safe schemas**: All data structures use Pydantic for validation
- **Configurable pipelines**: YAML configuration drives scraper selection and parameters

### Primary Entry Points
- `scripts/refresh_all_data.py` - **Main data collection pipeline**
- `scripts/refresh_core_dataset.py` - **Core dataset curation and management** 
- `scripts/query_models.py` - Interactive data exploration (optional)
- `scripts/manual_model_entry.py` - Manual model addition/correction
- `scripts/validate_against_epoch.py` - Cross-validation against Epoch's tracker

### Key Files
- `src/epoch_tracker/models/model.py` - Core data schemas
- `src/epoch_tracker/scrapers/base.py` - Base scraper interface
- `src/epoch_tracker/query/engine.py` - Query and filtering logic
- `src/epoch_tracker/estimation/compute_estimator.py` - FLOP calculation methods
- `configs/default.yaml` - System configuration

## Data Flow

1. **Collection**: Scrapers fetch data from various sources (HuggingFace, ChatBot Arena, etc.)
2. **Processing**: Raw data normalized to common schema and stored in `data/scraped/`
3. **Estimation**: FLOP estimates applied and models stored in `data/estimated/` 
4. **Curation**: Core dataset generated in `data/clean/` with manual verification workflow
5. **Querying**: Query engine filters and formats results for analysis

## Development Context

### Reference Documentation
- **Assignment.md**: Project objectives and success criteria
- **Plan.md**: Development roadmap and architecture decisions  
- **Notes.md**: Research references and informal notes (feel free to edit)
- **docs/**: Detailed methodology documentation

### Current System Status

The system is operational with the following capabilities:

**Data Sources:**
- LMArena: Manual CSV data + automated web scraping
- OpenLM Arena: Automated leaderboard data collection
- Papers with Code: Academic model releases (limited success)

**FLOP Estimation (Latest Run):**
- 418 total models processed
- 400 models with FLOP estimates
- 389 models above 1e25 FLOP threshold
- Methods: Known specs (15), benchmark interpolation (320), parameter heuristics (65)
- Confidence levels: High/Medium/Low/Speculative

**Core Dataset Management:**
- 400 candidate models for manual verification
- 15 models manually verified as above 1e25 FLOP
- Incremental updates preserve manual verification work
- CSV format for easy review and annotation

**Current Coverage:**
To check system status, run:
```bash
python scripts/refresh_all_data.py              # Full pipeline refresh
python scripts/refresh_core_dataset.py          # Update core dataset
python scripts/query_models.py --stats          # Show system statistics  
python scripts/query_models.py --above-threshold --format table  # View results
```

## Compute Estimation Methods

The system uses a hierarchical approach for FLOP estimation with the following priority order:

0. **Manual Overrides** (NEW) - Curated values from Epoch AI's authoritative tracker (HIGH/MEDIUM/LOW confidence)
1. **Known Model Specifications** - Official disclosures with published parameters/tokens (HIGH confidence)
2. **Parameter-based Chinchilla Scaling** - Extract params from names, intelligent token estimation (MEDIUM confidence)
3. **Benchmark-Based Interpolation** - ELO ratings → FLOP using reference models (MEDIUM/LOW confidence)  
4. **Hardware/Cost-Based** - GPU/TPU specs or training cost estimates (LOW confidence)

**Key Innovations:** 
- **Manual Override System**: Frontier models now use authoritative FLOP values from Epoch's research (Claude 3 Opus: 1.6e25, GPT-4: 2.1e25, Grok-2: 3.0e25)
- **Parameter-based Scaling**: Models with parameter counts in their names (e.g., `llama2_70b`, `qwen3_32b`) automatically use Chinchilla scaling laws with era-aware token estimation

See `docs/flop_estimation_guide.md` for detailed methodology documentation.

## Development Guidelines

**Architecture Principles:**
- **Python-first**: Modern Python with Pydantic validation and type hints
- **Modular design**: Extensible scrapers, pluggable estimators, configurable pipelines
- **Non-Epoch sources**: Primary data from HuggingFace, ChatBot Arena, company disclosures
- **Validation-ready**: Cross-validation against Epoch's existing tracker

**Key Considerations:**
- **FLOP Definition**: Primary focus on training compute >1e25 FLOP
- **Data Sources**: Prioritize public sources over proprietary benchmarks
- **Estimation Quality**: Multiple methods provide confidence bounds and validation
- **Reference Models**: Use well-characterized models (Llama 405B, Claude 3.5) as anchors

## Model Exclusion and Data Quality

The system includes automatic filtering to exclude low-quality or problematic model names during data collection. This prevents generic names like "gpt", "claude", or "test" from cluttering the dataset.

**Configuring Model Exclusions:**
- Edit the `EXCLUDED_MODEL_NAMES` set in `scripts/get_latest_model_data.py` (lines 40-64)
- Add model names that should be filtered out (case-insensitive matching)
- Examples: Generic names without versions, test models, placeholders, unclear metadata
- Changes take effect on next data collection run

**Current Exclusions Include:**
- Generic model families: "gpt", "claude", "llama", "gemini", "mistral" (without specific versions)
- Test/demo models: "test", "demo", "example", "placeholder", "model", "bot"  
- Unclear identifiers: "assistant", "ai", "chatbot", "unknown"

The filtering is applied after name normalization but before storage, with detailed logging of what was excluded and why.