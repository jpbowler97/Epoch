# Core Dataset Management

This guide explains the system for managing the curated dataset of AI models above the 1e25 FLOP threshold, including the design decisions, verification workflows, and technical implementation details.

## Overview

The core dataset management system addresses a critical challenge in AI model tracking: **how to maintain a high-quality, manually-verified dataset of models above the 1e25 FLOP threshold while incorporating automated updates from estimation algorithms**.

### Key Design Principles

1. **Hybrid Human-AI Curation**: Combines automated FLOP estimation with human verification
2. **Incremental Updates**: Preserves manual work while refreshing automated estimates
3. **Confidence-Based Filtering**: Focuses human attention on models most likely to exceed the threshold
4. **Audit Trail**: Maintains reasoning and source information for all entries

## System Architecture

### Data Flow

```
estimated_models.json (Stage 2) 
    ↓ [Candidate Filtering]
    ↓ [HIGH_CONFIDENCE_ABOVE + NOT_SURE classifications]
    ↓
data/clean/above_1e25_flop.csv (Stage 3)
    ↓ [Manual Verification]
    ↓ [verified=y marks confirmed entries]
    ↓
Epoch's Production Dataset
```

### File Structure

```
data/clean/above_1e25_flop.csv
├── Model metadata (name, developer, parameters, etc.)
├── FLOP estimates (training_flop, confidence, method, reasoning)
├── Threshold classification (high_confidence_above, not_sure)
├── Verification status (verified, notes)
└── Audit information (sources, last_updated)
```

## The `refresh_core_dataset.py` Script

### Core Logic

The script implements a **smart refresh algorithm** that balances automated updates with manual verification:

```python
def refresh_existing_csv(existing_df, models, csv_path):
    for row in existing_df:
        if row['verified'] == 'y':
            # Preserve manually verified entries unchanged
            keep_verified_entry(row)
        else:
            # Update with latest FLOP estimates
            update_from_latest_data(row, models)
    
    # Add new candidate models not in existing CSV
    add_new_candidates(models, existing_df)
```

### Candidate Model Selection

Models are included in the core dataset if they meet **either** criterion:

1. **HIGH_CONFIDENCE_ABOVE**: `training_flop >= 5e25` (5x the threshold)
   - Models we're confident exceed 1e25 FLOP
   - Typically from known model specifications or high-confidence estimations

2. **NOT_SURE**: `5e24 <= training_flop < 5e25` (uncertain range)
   - Models that might exceed 1e25 FLOP but need human verification
   - Includes benchmark-based estimates with uncertainty

**Excluded**: `HIGH_CONFIDENCE_BELOW` models (`training_flop <= 5e24`)
- Models we're confident are below the threshold

### Verification Workflow

#### Initial Dataset Creation

```bash
# First run creates CSV from scratch
python scripts/refresh_core_dataset.py
# → 400 candidate models, 0 verified
```

#### Manual Verification Process

1. **Review candidate models** in `data/clean/above_1e25_flop.csv`
2. **Research each model**: Check papers, blog posts, company announcements
3. **Mark verified entries**: Set `verified=y` for confirmed models
4. **Add notes**: Use `notes` column for verification reasoning

Example verification:
```csv
name,developer,...,verified,notes
gemini_2.5_pro,Google,...,y,Manually verified as above threshold based on Google AI blog post
claude_opus_4,Anthropic,...,y,Confirmed from Anthropic technical report
uncertain_model,Company,...,,Needs further research - conflicting estimates
```

#### Incremental Updates

```bash
# Subsequent runs preserve verified entries
python scripts/refresh_core_dataset.py
# → Updates 399 entries, preserves 1 verified entry
```

### Reset vs Refresh Modes

#### Refresh Mode (Default)
- **Preserves** entries where `verified=y`
- **Updates** non-verified entries with latest FLOP estimates
- **Adds** new candidate models
- **Keeps** manual notes and verification status

#### Reset Mode (`--reset`)
- **Recreates** dataset completely from scratch
- **Loses** all manual verification work
- **Use case**: Major changes to estimation methodology

```bash
# Careful: This discards all manual verification work
python scripts/refresh_core_dataset.py --reset
```

## Technical Implementation Details

### CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `name` | string | Model name |
| `developer` | string | Company/organization |
| `release_date` | ISO date | Model release date |
| `parameters` | integer | Parameter count |
| `training_flop` | float | Training FLOP estimate |
| `training_flop_formatted` | string | Human-readable FLOP (e.g., "5.54e+25") |
| `confidence` | enum | Estimation confidence (high/medium/low/speculative) |
| `estimation_method` | enum | Method used (scaling_laws/benchmark_based/etc.) |
| `threshold_classification` | enum | Threshold classification |
| `status` | enum | Model status (confirmed_above/likely_above/etc.) |
| `reasoning` | string | Detailed reasoning for FLOP estimate |
| `sources` | string | Semicolon-separated source URLs |
| `verified` | string | 'y' if manually verified, empty otherwise |
| `last_updated` | ISO datetime | Last update timestamp |
| `notes` | string | Manual verification notes |

### Deduplication Logic

Models are deduplicated by `(name.lower(), developer.lower())` with preference for:
1. Higher confidence estimates
2. More recent data

### Error Handling

The script includes robust error handling for common scenarios:
- Missing `estimated_models.json` file
- Corrupted CSV files
- Network connectivity issues
- Permission errors

## Best Practices

### For Researchers

1. **Verify in batches**: Focus on highest FLOP estimates first
2. **Document reasoning**: Use the `notes` field extensively
3. **Check multiple sources**: Don't rely on single data points
4. **Update regularly**: Run refresh weekly to catch new models

### For Developers

1. **Test with `--dry-run`**: Always test changes before applying
2. **Backup before reset**: Save manual work before using `--reset`
3. **Monitor logs**: Use `--verbose` for debugging
4. **Validate outputs**: Check row counts and verification statistics

### For System Administrators

1. **Automate data collection**: Set up cron jobs for stages 1-2
2. **Manual curation**: Stage 3 requires human oversight
3. **Version control**: Commit CSV files to track verification progress
4. **Access control**: Limit who can modify verified entries

## Monitoring and Quality Control

### Statistics Tracking

Each run provides detailed statistics:
```
CORE DATASET SUMMARY
Total candidate models: 400
Manually verified: 15
High confidence above 1e25: 8
Uncertain (need verification): 377
```

### Quality Metrics

- **Verification rate**: `verified_models / total_candidates`
- **Coverage rate**: `models_above_threshold / estimated_total`
- **Confidence distribution**: Breakdown by confidence levels
- **Update frequency**: How often estimates change

### Validation Checks

The system includes several validation mechanisms:
- Schema validation for all CSV fields
- FLOP value range checks (must be positive, reasonable magnitude)
- Date format validation
- Source URL validation
- Duplicate detection

## Troubleshooting

### Common Issues

**Missing estimated_models.json**
```bash
# Solution: Run FLOP estimation first
python scripts/estimate_flops.py --update
```

**CSV corruption**
```bash
# Solution: Reset from estimated data
python scripts/refresh_core_dataset.py --reset
# Then re-apply manual verification
```

**Large estimate changes**
- Check if estimation methodology changed
- Review models with significant FLOP changes
- Consider re-verification for affected models

### Debug Mode

```bash
# Detailed logging for troubleshooting
python scripts/refresh_core_dataset.py --verbose
```

## Future Enhancements

### Planned Features

1. **Version control integration**: Automatic commits for tracking changes
2. **Collaborative verification**: Multi-user workflow with conflict resolution
3. **External validation**: Cross-reference with Epoch's existing dataset
4. **Quality scoring**: Confidence metrics for verified entries
5. **Export formats**: Direct integration with Epoch's production systems

### API Extensions

The core dataset system is designed to support future API endpoints:
- `GET /api/models/above-threshold` - Retrieve verified models
- `POST /api/models/{id}/verify` - Mark model as verified
- `GET /api/models/statistics` - Dataset statistics and quality metrics

This comprehensive system ensures that Epoch maintains a high-quality, up-to-date dataset of models above the 1e25 FLOP threshold while maximizing the value of both automated estimation and human expertise.