# Staging/Published Dataset Synchronization

## Overview

The staging/published sync system enables bidirectional synchronization between the staging dataset (`data/staging/above_1e25_flop_staging.csv`) and the published dataset (`data/published/published_data.csv`). This allows for controlled promotion of models from staging to production and integration of manual corrections from the published dataset back into staging.

## Usage

```bash
python scripts/run.py sync-datasets [options]
```

### Command Options

- `--check-mappings` - Check for unmapped models without performing sync
- `--suggest-mappings` - Interactively review and approve mapping suggestions
- `--staging-to-published` - Sync from staging to published dataset
- `--published-to-staging` - Sync from published to staging dataset
- `--dry-run` - Show preview of changes without applying them
- `--diff` - Interactive field value comparison and resolution between datasets

## Manual Name Mapping System

### Configuration File

The sync system uses a manual mapping dictionary in `configs/model_name_mapping.yaml` to control how model names are translated between datasets:

```yaml
# Maps standardized staging names to human-readable published names
gpt_4o: "GPT-4o"
claude_3_5_sonnet: "Claude 3.5 Sonnet"
gemini_2.0_flash: "Gemini 2.0 Flash"
```

### Key Features

- **Manual control**: All name mappings must be explicitly defined in the YAML file
- **Unmapped detection**: System flags models without mappings and prevents sync until resolved
- **Bidirectional mapping**: Same configuration supports both sync directions
- **Schema translation**: Automatically handles field name differences between datasets

## Dataset Schema Differences

The two datasets use different schemas that are automatically translated during sync:

| Staging Field | Published Field | Notes |
|---------------|-----------------|-------|
| `model` | `Model` | Uses name mapping dictionary |
| `developer` | `Organization` | Direct translation |
| `training_flop` | `Training compute (FLOP)` | Direct translation |
| `parameters` | `Parameters` | Direct translation |
| `confidence` | `Confidence` | Direct translation |
| `reasoning` | `Training compute notes` | Direct translation |
| `release_date` | `Publication date` | Direct translation |
| `notes` | `Abstract` | Direct translation |

## Workflow

### 1. Check for Unmapped Models

Before performing any sync, check for models that lack name mappings:

```bash
python scripts/run.py sync-datasets --check-mappings
```

This will identify models in either dataset that don't have corresponding entries in the mapping file.

### 2. Add Missing Mappings

For any unmapped models, add entries to `configs/model_name_mapping.yaml`:

```yaml
# Add new mappings as needed
new_model_staging_name: "New Model Published Name"
```

### 3. Preview Changes (Dry Run)

Before applying changes, preview what will be modified:

```bash
# Preview staging → published sync
python scripts/run.py sync-datasets --dry-run --direction staging-to-published

# Preview published → staging sync  
python scripts/run.py sync-datasets --dry-run --direction published-to-staging
```

### 4. Execute Sync with Approval

Run the actual sync with interactive approval for each change:

```bash
# Sync staging → published
python scripts/run.py sync-datasets --sync --direction staging-to-published

# Sync published → staging
python scripts/run.py sync-datasets --sync --direction published-to-staging
```

### 5. Resolve Field Value Differences

Compare and resolve field value differences between existing models in both datasets:

```bash
# Interactive field comparison and resolution
python scripts/run.py sync-datasets --diff
```

This mode finds models that exist in both datasets and compares their field values. For each difference, you can choose to keep the staging value, published value, or skip the difference.

**🔒 Persistence Protection**: When you make changes via diff resolution, the affected models are automatically marked as `verified=y` in the staging dataset. This protects your manual corrections from being overwritten when automation scripts are rerun.

## Safety Features

### Mapping Enforcement
- **Pre-sync validation**: All sync operations check for unmapped models before proceeding
- **Clear warnings**: Shows exactly which models are unmapped and need attention
- **Guided resolution**: Suggests running `--suggest-mappings` to resolve issues
- **Override option**: Advanced users can proceed without mappings (not recommended)

### Additive-Only Operations
- **No deletions**: Sync never removes existing rows from target datasets
- **Updates only**: Only adds new models or updates existing ones with newer data
- **Preserve manual work**: Manual edits in target datasets are preserved

### Audit Trail
- **Staging → Published**: Adds timestamp comment to `Accessibility notes` field documenting automation pipeline source
- **Published → Staging**: Adds timestamp comment to `notes` field documenting automation pipeline source
- **Diff Resolution**: Adds timestamp comment to staging `notes` field documenting manual changes via diff resolution
- **Existing notes**: Audit comments are appended to existing notes, preserving original content

### Interactive Approval
For each model to be synced, the system displays:
- Model name and developer
- All field changes being made
- Source of the data (staging or published)

You must explicitly approve each individual sync operation.

### Unmapped Model Protection
- Sync is blocked if any models lack name mappings
- Prevents accidental data corruption from incorrect name translations
- Forces explicit mapping decisions for edge cases

## Example Session

### Checking for Unmapped Models

```bash
$ python scripts/run.py sync-datasets --check-mappings

Checking for unmapped models...

📋 Staging models without published mappings:
  - kimi_k2_preview
  - deepseek_r1_preview
  
📋 Published models without staging mappings:
  - "Claude Opus 4 (thinking 16k)" 
  - "Grok-4 Beta"

❌ Found 4 unmapped models. Add mappings to configs/model_name_mapping.yaml before syncing.
```

### Suggesting and Approving Mappings

```bash
$ python scripts/run.py sync-datasets --suggest-mappings

================================================================================
MAPPING SUGGESTIONS
================================================================================
Found 2 unmapped models

[1/2] Model: claude_ops_4
Developer: Anthropic
Suggested mapping: "Claude Opus 4"

Options: (a)ccept, (e)dit, (s)kip, (q)uit: a
✓ Approved: claude_ops_4 -> "Claude Opus 4"

[2/2] Model: deepseek_r1_preview
Developer: DeepSeek
Suggested mapping: "Deepseek R1 Preview"

Options: (a)ccept, (e)dit, (s)kip, (q)uit: e
Enter custom published name: DeepSeek R1 (preview)
✓ Custom mapping: deepseek_r1_preview -> "DeepSeek R1 (preview)"

================================================================================
Ready to add 2 mappings to configs/model_name_mapping.yaml
These will be organized by developer in the appropriate sections.

Proceed with adding to YAML file? (y/n): y

✓ Added 2 mappings to configs/model_name_mapping.yaml
```

The approved mappings are automatically added to the correct developer sections in the YAML file.

### Dry Run Preview

```bash
$ python scripts/run.py sync-datasets --dry-run --direction staging-to-published

🔍 DRY RUN: Staging → Published Sync Preview

📋 New models to add to published dataset:
┌─────────────────────────┬──────────────────────────┬─────────────────┐
│ Staging Name           │ Published Name           │ Training FLOP   │
├─────────────────────────┼──────────────────────────┼─────────────────┤
│ kimi_k2_preview        │ Kimi K2 (preview)        │ 4.55e+25       │
│ deepseek_r1           │ DeepSeek R1              │ 2.81e+25       │
└─────────────────────────┴──────────────────────────┴─────────────────┘

📋 Models to update in published dataset:
┌─────────────────────────┬──────────────────────────┬─────────────────┐
│ Staging Name           │ Published Name           │ Changes         │
├─────────────────────────┼──────────────────────────┼─────────────────┤
│ gpt_4o                 │ GPT-4o                   │ FLOP: 2.1e+25   │
└─────────────────────────┴──────────────────────────┴─────────────────┘

Total: 2 additions, 1 update
```

### Interactive Sync

```bash
$ python scripts/run.py sync-datasets --sync --direction staging-to-published

🔄 Staging → Published Sync

Processing 3 models for sync...

================================================================================
ADD: kimi_k2_preview → "Kimi K2 (preview)"
================================================================================
Organization: Moonshot
Training compute (FLOP): 4.55e+25
Confidence: medium
Training compute notes: Benchmark-based estimation from OpenLM Arena ELO scores
Publication date: 2024-07-11
Abstract: Large frontier model from Moonshot AI with strong benchmark performance
Accessibility notes: Added via automation pipeline from staging dataset on 2025-08-01 14:30 UTC

Approve this addition? (y/n): y
✅ Added "Kimi K2 (preview)" to published dataset

================================================================================
UPDATE: gpt_4o → "GPT-4o"  
================================================================================
Changes:
  Training compute (FLOP): 1.8e+25 → 2.1e+25
  Training compute notes: Updated with latest Epoch AI research data

Approve this update? (y/n): y  
✅ Updated "GPT-4o" in published dataset

Sync completed: 2 additions, 1 update
📁 Published dataset saved to: data/published/published_data.csv
```

### Field Value Diff Resolution

```bash
$ python scripts/run.py sync-datasets --diff

📋 INTERACTIVE FIELD DIFF RESOLUTION

Found 4 models in both datasets

Options for each difference:
  p - Keep published value (update staging)
  s - Keep staging value (update published)
  skip - Skip this difference
  quit - Exit diff resolution

================================================================================
MODEL 1/4: gpt_4.5_preview ↔ GPT-4.5
================================================================================
Current status: ⚠ UNVERIFIED
📝 Note: Selecting 'p' or 's' will mark this model as verified (protected from automation overwrites)
Found 3 field differences

[DIFF 1/3] Field: training_flop ↔ Training compute (FLOP)
  Staging:   6.40e+25
  Published: 2.10e+26
Resolution (p/s/skip/quit): p
  ✓ Updated staging training_flop to: 2.10e+26
  ✓ Marked gpt_4.5_preview as verified (protected from automation overwrites)

[DIFF 2/3] Field: confidence ↔ Confidence
  Staging:   medium
  Published: High
Resolution (p/s/skip/quit): p
  ✓ Updated staging confidence to: High
  ✓ Marked gpt_4.5_preview as verified (protected from automation overwrites)

[DIFF 3/3] Field: reasoning ↔ Training compute notes
  Staging:   Benchmark-based estimation
  Published: Official OpenAI disclosure
Resolution (p/s/skip/quit): p
  ✓ Updated staging reasoning to: Official OpenAI disclosure
  ✓ Marked gpt_4.5_preview as verified (protected from automation overwrites)

================================================================================
DIFF RESOLUTION COMPLETE
================================================================================
Total updates made: 3

✓ Updated staging dataset saved to: data/staging/above_1e25_flop_staging.csv
```

## Integration with Pipeline

The sync system integrates with the broader data pipeline:

1. **Stage 1-4**: Run normal pipeline through manual review
2. **Staging ready**: Models accumulate in `data/staging/above_1e25_flop_staging.csv`
3. **Production sync**: Use sync system to promote models to published dataset
4. **Manual corrections**: Edit published dataset directly for corrections
5. **Reverse sync**: Pull manual corrections back to staging for consistency

## Configuration Management

### Adding New Models

When new models are identified that need mapping:

1. Add entry to `configs/model_name_mapping.yaml`
2. Follow naming conventions (staging: `snake_case`, published: `"Human Readable"`)
3. Test mapping with `--check-mappings` before syncing

### Mapping Guidelines

- **Staging names**: Use normalized snake_case format (e.g., `claude_3_5_sonnet`)
- **Published names**: Use human-readable format with proper capitalization (e.g., `"Claude 3.5 Sonnet"`)
- **Version handling**: Include version numbers and variants as needed
- **Consistency**: Maintain consistent naming patterns within model families

## Troubleshooting

### Common Issues

1. **Unmapped models**: Add missing entries to mapping file
2. **Schema mismatches**: Verify field mappings in sync script
3. **Encoding issues**: Ensure YAML file uses UTF-8 encoding
4. **Merge conflicts**: Resolve manually in target dataset before retrying

### Validation

After sync operations, validate results:

```bash
# Check row counts
wc -l data/staging/above_1e25_flop_staging.csv
wc -l data/published/published_data.csv

# Verify no duplicates
python scripts/run.py query --check-duplicates
```