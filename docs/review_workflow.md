# Model Review Workflow

## Overview

The semi-automatic model review process allows you to compare models in `clean/above_1e25_flop.csv` against the staging dataset and make manual decisions about each new candidate.

## Usage

```bash
python scripts/run.py review-candidates [--data-dir DATA_DIR] [--batch-size BATCH_SIZE]
```

### Options

- `--data-dir`: Base data directory (default: `data`)
- `--batch-size`: Number of models to review before asking to continue (default: 10)

## Workflow

1. **Load datasets**: The script loads candidate models from `clean/above_1e25_flop.csv` and the staging dataset from `staging/above_1e25_flop_staging.csv`

2. **Find new candidates**: Models that exist in candidates but not in staging are identified for review

3. **Interactive review**: For each new candidate, the script displays:
   - Model name and developer
   - Training FLOP estimate and confidence
   - Estimation method and reasoning
   - Sources and alternative methods
   - All other relevant metadata

4. **User decisions**: For each model, you can respond with:
   - `y` - Add the model to the staging dataset (confirming it's above 1e25 FLOP) - **prompts for reason**
   - `n` - Move the model to `below_1e25_flop.csv` with `verified=y` (confirming it's below threshold) - **prompts for reason**
   - `skip` - Skip this model for now (can review later)
   - `quit` - Exit the review process

## Key Features

### Reason Tracking
When you choose `y` or `n` for a model, you'll be prompted to provide a reason for your decision. This reason is saved in the `notes` field for future reference and audit purposes.

### Verified Protection
When you choose `n` for a model:
- The model is moved from `clean/above_1e25_flop.csv` to `clean/below_1e25_flop.csv`
- The `verified` field is set to `y`
- The `status` is updated to `confirmed_below_1e25`
- The `threshold_classification` is set to `high_confidence_below_1e25`
- Your reason is added to the `notes` field

When you choose `y` for a model:
- The model is added to the staging dataset with your reason in the `notes` field
- The model is marked as `verified=y` in the original `clean/above_1e25_flop.csv`
- Your reason is also added to the `notes` field in the candidates file

**Important**: Models with `verified=y` are protected from automatic pipeline updates. Even if the automated pipeline later estimates different FLOP values for these models, they will remain in their designated files and will not be moved by automated processes.

### Staging Dataset Management
The staging dataset (`staging/above_1e25_flop_staging.csv`) contains all models that have been manually confirmed as above 1e25 FLOP. This dataset is ready for production deployment to Airtable or other downstream systems.

## Example Session

```
Loading datasets...
Candidates: 123 models
Staging: 19 models

Found 104 new candidates to review

Options for each model:
  y - Add to staging dataset (model is above 1e25 FLOP) - will prompt for reason
  n - Move to below_1e25_flop.csv with verified=y (model is below threshold) - will prompt for reason
  skip - Skip this model for now
  quit - Exit the review process

================================================================================
MODEL: kimi_k2_preview
================================================================================
Developer: Moonshot
Training FLOP: 4.55e+25
Confidence: medium
Estimation Method: benchmark_based
Threshold Classification: not_sure
Status: likely_above_1e25

Reasoning: Benchmark-based (openlm_arena_elo): Benchmark-based estimation...

Sources: https://openlm.ai/chatbot-arena/ (OpenLM Chatbot Arena leaderboard)
================================================================================

Decision (y/n/skip/quit): y
Please provide a reason for confirming this model is above 1e25 FLOP: Strong benchmark performance and known to be a large frontier model from Moonshot AI

Successfully added kimi_k2_preview to staging dataset and marked as verified
```

## Integration with Pipeline

This workflow integrates seamlessly with the existing automated pipeline:

1. Run the automated pipeline: `python scripts/run.py collect-all && python scripts/run.py estimate-flops --update`
2. Review new candidates: `python scripts/run.py review-candidates`
3. Re-run the core dataset refresh: `python scripts/run.py refresh-dataset`

The verified models will be protected from future automated changes, ensuring your manual decisions are preserved.