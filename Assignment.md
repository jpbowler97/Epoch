# Assignment: Improve Epoch's Tracking of >1e25 FLOP AI Models

**Objective:**  
Design and implement a robust, semi-automated system to improve Epoch’s tracking of AI models trained with over 1e25 floating-point operations (FLOP). The system should support better schema definition, reasoning transparency, partial automation, and ease of manual updates.

**Context:**  
Epoch currently tracks large-scale AI models (e.g., GPT-4 and beyond) at:  
https://epoch.ai/data-insights/models-over-1e25-flop

The current process is largely ad hoc. Models are identified from:
- AI lab releases
- Benchmark and leaderboard results
- Model repositories

Compute estimates are made based on available information, and models are classified as likely above or below the 1e25 FLOP threshold. Much of this work is manual and hard to delegate.

**Goal:**  
Build a pipeline and data structure that:
- Tracks the model’s metadata and compute status in a structured schema
- Makes the reasoning behind inclusion/exclusion transparent
- Automates parts of model discovery and estimation (e.g. leaderboard scraping)
- Enables easy, low-friction manual updates via contractor-friendly workflows

The ideal system would allow the insight to stay current by default, requiring only lightweight, periodic updates.

**Deliverables:**
- **Schema for model metadata and status** - `src/epoch_tracker/models/model.py`
- **Scripts for data ingestion and compute estimation** - `scripts/get_latest_model_data.py`
- **Interface for data access** - `scripts/query_models.py` CLI tool
- **Documentation for maintainers and contributors** - See documentation links below

## What does success look like?

Being able to replicate the "Data" section here https://epoch.ai/data-insights/models-over-1e25-flop by automatically scraping the information from publicly available sources.

Then have the ability to manually intervene to correct issues / add more models. This manual process should be well defined and manageable by a contractor.