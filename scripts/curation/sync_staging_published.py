#!/usr/bin/env python3
"""
Synchronization tool for staging and published datasets.

This script provides safe, manual-controlled synchronization between:
- staging/above_1e25_flop_staging.csv (standardized names, technical fields)
- published/published_data.csv (human-readable names, publication metadata)

Key Features:
- Manual name mapping control via configs/model_name_mapping.yaml
- Additive-only sync (never removes existing data)
- Interactive approval for each proposed change
- Unmapped model detection and flagging
- Bidirectional sync capability with safety checks
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import yaml
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# SCHEMA TRANSLATION MAPS
# =============================================================================

# Primary mapping: staging field names → published field names
# This is the single source of truth for field mappings
STAGING_TO_PUBLISHED_FIELD_MAP = {
    'model': 'Model',  # Special handling - uses name mapping
    'developer': 'Organization',
    'training_flop': 'Training compute (FLOP)', 
    'parameters': 'Parameters',
    'confidence': 'Confidence',
    'reasoning': 'Training compute notes',
    'release_date': 'Publication date',
    'notes': 'Accessibility notes',
    'sources': 'Link',
    'estimation_method': 'Training compute notes',  # Append to existing notes
    'verified': None,  # No published equivalent
    'last_updated': None,  # No published equivalent
    'threshold_classification': None,  # No published equivalent
    'status': None,  # No published equivalent
    'alternative_methods': None,  # No published equivalent
    'training_flop_formatted': None,  # No published equivalent
}

# Published fields that don't have staging equivalents
PUBLISHED_ONLY_FIELDS = {
    'Domain', 'Task', 'Authors', 'Model accessibility', 'Citations', 'Reference',
    'Parameters notes', 'Training dataset', 'Training dataset notes', 
    'Training dataset size (datapoints)', 'Dataset size notes', 'Training time (hours)',
    'Training time notes', 'Training hardware', 'Country (of organization)',
    'Base model', 'Finetune compute (FLOP)', 'Finetune compute notes',
    'Hardware quantity', 'Training code accessibility', 'Abstract',
    'Organization categorization (from Organization)'
}


def get_staging_to_published_map() -> Dict[str, str]:
    """Get the staging→published field mapping."""
    return STAGING_TO_PUBLISHED_FIELD_MAP.copy()


def get_published_to_staging_map() -> Dict[str, str]:
    """Generate the published→staging field mapping by reversing the primary mapping."""
    reverse_map = {}
    
    # Add reverse mappings for fields that have equivalents
    for staging_field, published_field in STAGING_TO_PUBLISHED_FIELD_MAP.items():
        if published_field is not None:
            reverse_map[published_field] = staging_field
    
    # Add published-only fields (map to None)
    for published_field in PUBLISHED_ONLY_FIELDS:
        reverse_map[published_field] = None
    
    return reverse_map


# =============================================================================
# MAPPING FILE MANAGEMENT
# =============================================================================

def load_name_mapping(mapping_file: Path) -> Dict[str, str]:
    """Load the manual name mapping configuration.
    
    Args:
        mapping_file: Path to model_name_mapping.yaml
        
    Returns:
        Dictionary mapping staging names to published names
        
    Raises:
        FileNotFoundError: If mapping file doesn't exist
        yaml.YAMLError: If mapping file is invalid YAML
    """
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_file}\n"
            f"Please create the mapping file with initial model name mappings."
        )
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in mapping file {mapping_file}: {e}")
    
    # Validate mapping structure
    if not isinstance(mapping, dict):
        raise ValueError(f"Mapping file must contain a dictionary, got {type(mapping)}")
    
    # Filter out comments and ensure all values are strings
    validated_mapping = {}
    for key, value in mapping.items():
        if isinstance(key, str) and isinstance(value, str):
            validated_mapping[key] = value
        elif key is not None and value is not None:
            print(f"Warning: Skipping invalid mapping entry: {key} -> {value}")
    
    return validated_mapping


def get_reverse_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """Create reverse mapping from published names to staging names.
    
    Args:
        mapping: Staging name -> published name mapping
        
    Returns:
        Published name -> staging name mapping
    """
    return {v: k for k, v in mapping.items()}


def add_mappings_to_yaml(approved_mappings: Dict[str, Tuple[str, str]], mapping_file: Path) -> None:
    """Add approved mappings to the main YAML file, organized by developer.
    
    Args:
        approved_mappings: Dict of staging_name -> (developer, published_name)
        mapping_file: Path to model_name_mapping.yaml
    """
    # Read existing file content
    with open(mapping_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Group approved mappings by developer
    mappings_by_developer = {}
    for staging_name, (developer, published_name) in approved_mappings.items():
        if developer not in mappings_by_developer:
            mappings_by_developer[developer] = []
        mappings_by_developer[developer].append((staging_name, published_name))
    
    # Sort mappings within each developer group
    for developer in mappings_by_developer:
        mappings_by_developer[developer].sort(key=lambda x: x[0])
    
    # Process each developer's mappings
    for developer, mappings in mappings_by_developer.items():
        # Find the section for this developer
        section_header = f"# {developer} Models"
        section_found = False
        insert_index = -1
        
        for i, line in enumerate(lines):
            if line.strip() == section_header:
                section_found = True
                # Find the end of this section (next comment or EOF)
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith('#') or j == len(lines) - 1:
                        insert_index = j
                        if j == len(lines) - 1 and not lines[j].strip().startswith('#'):
                            insert_index = j + 1
                        break
                break
        
        # If section not found, create it at the end
        if not section_found:
            # Add new section at the end (before final comments if any)
            insert_index = len(lines)
            # Add spacing and section header
            if lines and not lines[-1].strip() == '':
                lines.append('\n')
            lines.append(f'{section_header}\n')
            insert_index = len(lines)
        
        # Insert the new mappings
        for staging_name, published_name in reversed(mappings):
            mapping_line = f'{staging_name}: "{published_name}"\n'
            lines.insert(insert_index, mapping_line)
    
    # Write updated content back to file
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"\n✓ Added {len(approved_mappings)} mappings to {mapping_file}")


# =============================================================================
# UNMAPPED MODEL DETECTION
# =============================================================================

def detect_unmapped_models(
    staging_df: pd.DataFrame, 
    published_df: pd.DataFrame, 
    mapping: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """Detect models that don't have mapping entries.
    
    Args:
        staging_df: Staging dataset DataFrame
        published_df: Published dataset DataFrame  
        mapping: Staging name -> published name mapping
        
    Returns:
        Tuple of (unmapped_staging_models, unmapped_published_models)
    """
    reverse_mapping = get_reverse_mapping(mapping)
    
    # Find staging models without mappings
    unmapped_staging = []
    if 'model' in staging_df.columns:
        for model in staging_df['model'].dropna().unique():
            if str(model) not in mapping:
                unmapped_staging.append(str(model))
    
    # Find published models without reverse mappings
    unmapped_published = []
    if 'Model' in published_df.columns:
        for model in published_df['Model'].dropna().unique():
            if str(model) not in reverse_mapping:
                unmapped_published.append(str(model))
    
    return unmapped_staging, unmapped_published


def suggest_mappings(unmapped_staging: List[str]) -> Dict[str, str]:
    """Suggest published names for unmapped staging models.
    
    Args:
        unmapped_staging: List of staging model names without mappings
        
    Returns:
        Dictionary of suggested mappings
    """
    suggestions = {}
    
    for staging_name in unmapped_staging:
        # Use existing normalization logic in reverse to suggest human-readable name
        suggested_name = staging_name_to_human_readable(staging_name)
        suggestions[staging_name] = suggested_name
    
    return suggestions


def staging_name_to_human_readable(staging_name: str) -> str:
    """Convert standardized staging name to human-readable format.
    
    Args:
        staging_name: Standardized model name (e.g., 'gpt_4.5_preview')
        
    Returns:
        Human-readable name (e.g., 'GPT-4.5 Preview')
    """
    if not staging_name:
        return staging_name
    
    # Split by underscores and capitalize appropriately
    parts = staging_name.split('_')
    result_parts = []
    
    for part in parts:
        # Handle known patterns
        if part.lower() in ['gpt', 'api', 'ai', 'llm', 'pro', 'mini', 'ultra', 'lite']:
            result_parts.append(part.upper())
        elif part.lower() in ['claude', 'gemini', 'grok', 'llama', 'deepseek', 'mistral']:
            result_parts.append(part.capitalize())
        elif part.lower() in ['opus', 'sonnet', 'haiku', 'flash', 'preview', 'turbo']:
            result_parts.append(part.capitalize())
        elif part.lower() == 'ops':
            # Fix common typo: ops -> Opus
            result_parts.append('Opus')
        elif part.lower() in ['thinking', 'coder', 'instruct', 'base', 'chat']:
            result_parts.append(f"({part})")
        elif part.endswith('b') and part[:-1].isdigit():
            # Handle parameter counts like '405b', '70b'
            result_parts.append(part.upper())
        elif 'k' in part and any(c.isdigit() for c in part):
            # Handle context lengths like '16k', '32k'
            result_parts.append(f"({part})")
        elif part.replace('.', '').isdigit():
            # Handle version numbers
            result_parts.append(part)
        else:
            result_parts.append(part.capitalize())
    
    return ' '.join(result_parts)


def interactive_mapping_approval(
    suggestions: Dict[str, str],
    staging_df: pd.DataFrame,
    mapping_file: Path
) -> None:
    """Interactively approve mapping suggestions and add to YAML file.
    
    Args:
        suggestions: Dict of staging_name -> suggested_published_name
        staging_df: Staging dataframe to get developer info
        mapping_file: Path to model_name_mapping.yaml
    """
    if not suggestions:
        print("No unmapped models found.")
        return
    
    approved_mappings = {}
    
    print(f"\n{'='*80}")
    print("MAPPING SUGGESTIONS")
    print(f"{'='*80}")
    print(f"Found {len(suggestions)} unmapped models\n")
    
    for i, (staging_name, suggested_name) in enumerate(suggestions.items(), 1):
        # Get developer from staging dataset
        try:
            model_row = staging_df[staging_df['model'] == staging_name].iloc[0]
            developer = model_row.get('developer', 'Unknown')
        except:
            developer = 'Unknown'
        
        print(f"\n[{i}/{len(suggestions)}] Model: {staging_name}")
        print(f"Developer: {developer}")
        print(f"Suggested mapping: \"{suggested_name}\"")
        
        while True:
            choice = input("\nOptions: (a)ccept, (e)dit, (s)kip, (q)uit: ").strip().lower()
            
            if choice == 'q':
                print("\nQuitting mapping approval process.")
                break
                
            elif choice == 's':
                print("Skipped.")
                break
                
            elif choice == 'a':
                approved_mappings[staging_name] = (developer, suggested_name)
                print(f"✓ Approved: {staging_name} -> \"{suggested_name}\"")
                break
                
            elif choice == 'e':
                custom_name = input("Enter custom published name: ").strip()
                if custom_name:
                    approved_mappings[staging_name] = (developer, custom_name)
                    print(f"✓ Custom mapping: {staging_name} -> \"{custom_name}\"")
                    break
                else:
                    print("Invalid input. Please try again.")
            else:
                print("Invalid choice. Use 'a', 'e', 's', or 'q'.")
        
        if choice == 'q':
            break
    
    # If any mappings were approved, add them to the YAML file
    if approved_mappings:
        print(f"\n{'='*80}")
        print(f"Ready to add {len(approved_mappings)} mappings to {mapping_file}")
        print("These will be organized by developer in the appropriate sections.")
        
        confirm = input("\nProceed with adding to YAML file? (y/n): ").strip().lower()
        if confirm == 'y':
            add_mappings_to_yaml(approved_mappings, mapping_file)
        else:
            print("Cancelled. No changes made to mapping file.")
    else:
        print("\nNo mappings approved.")


# =============================================================================
# SCHEMA TRANSLATION
# =============================================================================

def translate_staging_to_published(staging_row: pd.Series, mapping: Dict[str, str]) -> Dict[str, Any]:
    """Translate a staging model row to published format.
    
    Args:
        staging_row: Row from staging DataFrame
        mapping: Staging name -> published name mapping
        
    Returns:
        Dictionary with published field names and values
    """
    published_row = {}
    
    # Handle model name mapping specially
    staging_name = str(staging_row.get('model', ''))
    if staging_name in mapping:
        published_row['Model'] = mapping[staging_name]
    else:
        raise ValueError(f"No mapping found for staging model: {staging_name}")
    
    # Translate other fields (skip 'model' field as it's handled specially above)
    for staging_field, published_field in get_staging_to_published_map().items():
        if published_field and staging_field != 'model' and staging_field in staging_row:
            value = staging_row[staging_field]
            if pd.notna(value):
                published_row[published_field] = value
    
    # Set default values for required published fields
    published_row.setdefault('Domain', 'Language')
    published_row.setdefault('Task', 'Language modeling/generation')
    published_row.setdefault('Model accessibility', 'API access')
    published_row.setdefault('Country (of organization)', 'Unknown')
    published_row.setdefault('Organization categorization (from Organization)', 'Industry')
    
    # Add automation pipeline audit comment
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    audit_comment = f"Added via automation pipeline from staging dataset on {timestamp}"
    
    # Add to Accessibility notes field
    existing_notes = published_row.get('Accessibility notes', '')
    if existing_notes and pd.notna(existing_notes) and str(existing_notes).strip():
        published_row['Accessibility notes'] = f"{existing_notes}. {audit_comment}"
    else:
        published_row['Accessibility notes'] = audit_comment
    
    return published_row


def translate_published_to_staging(published_row: pd.Series, reverse_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Translate a published model row to staging format.
    
    Args:
        published_row: Row from published DataFrame
        reverse_mapping: Published name -> staging name mapping
        
    Returns:
        Dictionary with staging field names and values
    """
    staging_row = {}
    
    # Handle model name mapping specially
    published_name = str(published_row.get('Model', ''))
    if published_name in reverse_mapping:
        staging_row['model'] = reverse_mapping[published_name]
    else:
        raise ValueError(f"No reverse mapping found for published model: {published_name}")
    
    # Translate other fields (skip 'Model' field as it's handled specially above)
    for published_field, staging_field in get_published_to_staging_map().items():
        if staging_field and published_field != 'Model' and published_field in published_row:
            value = published_row[published_field]
            if pd.notna(value):
                staging_row[staging_field] = value
    
    # Set default values for required staging fields
    staging_row.setdefault('verified', '')
    staging_row.setdefault('last_updated', datetime.now(timezone.utc).isoformat())
    staging_row.setdefault('threshold_classification', 'manual_entry')
    staging_row.setdefault('status', 'manual_import')
    staging_row.setdefault('estimation_method', 'manual_entry')
    staging_row.setdefault('confidence', 'unknown')
    
    # Add automation pipeline audit comment
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    audit_comment = f"Added via automation pipeline from published dataset on {timestamp}"
    
    # Add to notes field
    existing_notes = staging_row.get('notes', '')
    if existing_notes and pd.notna(existing_notes) and str(existing_notes).strip():
        staging_row['notes'] = f"{existing_notes}. {audit_comment}"
    else:
        staging_row['notes'] = audit_comment
    
    return staging_row


# =============================================================================
# SYNC OPERATIONS
# =============================================================================

def find_models_to_sync(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame, 
    mapping: Dict[str, str],
    direction: str
) -> List[str]:
    """Find models in source that don't exist in target dataset.
    
    Args:
        source_df: Source dataset DataFrame
        target_df: Target dataset DataFrame
        mapping: Name mapping dictionary
        direction: 'staging_to_published' or 'published_to_staging'
        
    Returns:
        List of model names to sync
    """
    models_to_sync = []
    
    if direction == 'staging_to_published':
        # Find staging models not in published
        staging_models = set(source_df['model'].dropna().astype(str))
        published_models = set(target_df['Model'].dropna().astype(str))
        
        for staging_model in staging_models:
            if staging_model in mapping:
                published_name = mapping[staging_model]
                if published_name not in published_models:
                    models_to_sync.append(staging_model)
    
    elif direction == 'published_to_staging':
        # Find published models not in staging
        reverse_mapping = get_reverse_mapping(mapping)
        published_models = set(source_df['Model'].dropna().astype(str))
        staging_models = set(target_df['model'].dropna().astype(str))
        
        for published_model in published_models:
            if published_model in reverse_mapping:
                staging_name = reverse_mapping[published_model]
                if staging_name not in staging_models:
                    models_to_sync.append(published_model)
    
    return models_to_sync


def display_model_details(model_name: str, df: pd.DataFrame, model_field: str) -> None:
    """Display detailed information about a model.
    
    Args:
        model_name: Name of the model to display
        df: DataFrame containing the model
        model_field: Field name containing model names
    """
    model_row = df[df[model_field] == model_name].iloc[0]
    
    print(f"Model: {model_name}")
    
    # Show key fields based on dataset type
    if model_field == 'model':  # Staging dataset
        key_fields = [
            ('Developer', 'developer'),
            ('Training FLOP', 'training_flop'),
            ('Parameters', 'parameters'),
            ('Confidence', 'confidence'),
            ('Estimation Method', 'estimation_method'),
            ('Reasoning', 'reasoning'),
            ('Sources', 'sources'),
            ('Verified', 'verified'),
            ('Last Updated', 'last_updated'),
        ]
    else:  # Published dataset  
        key_fields = [
            ('Organization', 'Organization'),
            ('Training Compute (FLOP)', 'Training compute (FLOP)'),
            ('Parameters', 'Parameters'),
            ('Confidence', 'Confidence'),
            ('Publication Date', 'Publication date'),
            ('Abstract', 'Abstract'),
            ('Link', 'Link'),
        ]
    
    for label, field in key_fields:
        if field in model_row and pd.notna(model_row[field]):
            value = model_row[field]
            if isinstance(value, float) and value > 1e20:
                print(f"  {label}: {value:.2e}")
            else:
                print(f"  {label}: {value}")


def display_translation_preview(
    model_name: str, 
    source_df: pd.DataFrame, 
    mapping: Dict[str, str],
    direction: str
) -> Dict[str, Any]:
    """Display what the translated model will look like.
    
    Args:
        model_name: Name of model being translated
        source_df: Source DataFrame
        mapping: Name mapping dictionary
        direction: Translation direction
        
    Returns:
        Translated model data
    """
    if direction == 'staging_to_published':
        source_row = source_df[source_df['model'] == model_name].iloc[0]
        translated = translate_staging_to_published(source_row, mapping)
        
        print(f"Will create published model: {translated['Model']}")
        key_fields = [
            ('Organization', 'Organization'),
            ('Training Compute (FLOP)', 'Training compute (FLOP)'),
            ('Parameters', 'Parameters'),
            ('Confidence', 'Confidence'),
            ('Training compute notes', 'Training compute notes'),
            ('Publication date', 'Publication date'),
            ('Abstract', 'Abstract'),
            ('Accessibility notes', 'Accessibility notes'),
            ('Link', 'Link'),
        ]
        
    else:  # published_to_staging
        reverse_mapping = get_reverse_mapping(mapping)
        source_row = source_df[source_df['Model'] == model_name].iloc[0]
        translated = translate_published_to_staging(source_row, reverse_mapping)
        
        print(f"Will create staging model: {translated['model']}")
        key_fields = [
            ('Developer', 'developer'),
            ('Training FLOP', 'training_flop'),
            ('Parameters', 'parameters'),
            ('Confidence', 'confidence'),
            ('Reasoning', 'reasoning'),
            ('Notes', 'notes'),
        ]
    
    for label, field in key_fields:
        if field in translated and translated[field] is not None:
            value = translated[field]
            if isinstance(value, float) and value > 1e20:
                print(f"  {label}: {value:.2e}")
            else:
                print(f"  {label}: {value}")
    
    return translated


def interactive_sync_approval(
    models_to_sync: List[str],
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    mapping: Dict[str, str],
    direction: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Interactive approval process for sync operations.
    
    Args:
        models_to_sync: List of models to sync
        source_df: Source DataFrame
        target_df: Target DataFrame  
        mapping: Name mapping dictionary
        direction: Sync direction
        
    Returns:
        Tuple of (approved_translated_models, models_to_remove_from_staging)
    """
    approved_models = []
    removed_models = []
    
    if not models_to_sync:
        print("No models to sync.")
        return approved_models, removed_models
    
    source_field = 'model' if direction == 'staging_to_published' else 'Model'
    
    print(f"\nFound {len(models_to_sync)} models to sync ({direction})")
    print("\nOptions for each model:")
    print("  y - Approve sync")
    print("  n - Skip this model")
    if direction == 'staging_to_published':
        print("  r - Remove this model from staging (don't sync)")
    print("  quit - Exit sync process")
    
    for i, model_name in enumerate(models_to_sync):
        print(f"\n{'='*80}")
        print(f"SYNC PROPOSAL {i+1}/{len(models_to_sync)}: {direction}")
        print(f"{'='*80}")
        
        print("\nSOURCE MODEL:")
        display_model_details(model_name, source_df, source_field)
        
        print("\nWILL CREATE:")
        try:
            translated = display_translation_preview(model_name, source_df, mapping, direction)
        except Exception as e:
            print(f"Error translating model: {e}")
            continue
        
        while True:
            if direction == 'staging_to_published':
                response = input("\nYour choice (y/n/r/quit): ").strip().lower()
            else:
                response = input("\nApprove this sync? (y/n/quit): ").strip().lower()
            
            if response == 'y':
                approved_models.append(translated)
                print(f"✓ Approved: {model_name}")
                break
            elif response == 'n':
                print(f"⏭️  Skipped: {model_name}")
                break
            elif response == 'r' and direction == 'staging_to_published':
                removed_models.append(model_name)
                print(f"🗑️  Model '{model_name}' will be removed from staging")
                break
            elif response == 'quit':
                print("Exiting sync process...")
                return approved_models, removed_models
            else:
                valid_options = "'y', 'n', 'quit'"
                if direction == 'staging_to_published':
                    valid_options = "'y', 'n', 'r', 'quit'"
                print(f"Invalid response. Please enter {valid_options}")
    
    return approved_models, removed_models


def perform_sync(
    approved_models: List[Dict[str, Any]],
    target_df: pd.DataFrame,
    target_path: Path,
    direction: str
) -> bool:
    """Perform the actual sync operation by adding approved models.
    
    Args:
        approved_models: List of translated models to add
        target_df: Target DataFrame to update
        target_path: Path to target CSV file
        direction: Sync direction for logging
        
    Returns:
        True if successful, False otherwise
    """
    if not approved_models:
        print("No approved models to sync.")
        return True
    
    try:
        # Create new rows DataFrame
        new_rows_df = pd.DataFrame(approved_models)
        
        # Ensure columns match target DataFrame
        for col in target_df.columns:
            if col not in new_rows_df.columns:
                new_rows_df[col] = None
        
        # Reorder columns to match target
        new_rows_df = new_rows_df[target_df.columns]
        
        # Add new rows to target DataFrame
        updated_df = pd.concat([target_df, new_rows_df], ignore_index=True)
        
        # Save updated DataFrame
        updated_df.to_csv(target_path, index=False)
        
        print(f"\n✓ Successfully synced {len(approved_models)} models to {target_path}")
        return True
        
    except Exception as e:
        print(f"\nError performing sync: {e}")
        return False


def interactive_diff_resolution(
    staging_df: pd.DataFrame,
    published_df: pd.DataFrame,
    mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Interactively compare and resolve field value differences between datasets.
    
    Args:
        staging_df: Staging dataset
        published_df: Published dataset
        mapping: Model name mapping dictionary
        
    Returns:
        Tuple of (updated_staging_df, updated_published_df)
    """
    staging_to_published_map = get_staging_to_published_map()
    
    # Find models that exist in both datasets
    common_models = []
    
    for staging_name in staging_df['model']:
        if staging_name in mapping:
            published_name = mapping[staging_name]
            if published_name in published_df['Model'].values:
                common_models.append((staging_name, published_name))
    
    if not common_models:
        print("No common models found between datasets for comparison.")
        return staging_df, published_df
    
    print(f"\n{'='*80}")
    print("INTERACTIVE FIELD VALUE COMPARISON")
    print(f"{'='*80}")
    print(f"Found {len(common_models)} models in both datasets")
    print("\nOptions for each difference:")
    print("  p - Keep published value (update staging)")
    print("  s - Keep staging value (update published)")
    print("  skip - Skip this difference")
    print("  quit - Exit diff resolution")
    print("\n💡 Models will be marked as 'verified' when updated, protecting them from automation overwrites")
    
    updates_made = 0
    
    for i, (staging_name, published_name) in enumerate(common_models, 1):
        staging_row = staging_df[staging_df['model'] == staging_name].iloc[0]
        published_row = published_df[published_df['Model'] == published_name].iloc[0]
        
        # Find field differences
        differences = []
        
        # Check mappable fields (skip 'model' field since name differences are expected)
        for staging_field, published_field in staging_to_published_map.items():
            if (published_field and staging_field != 'model' and 
                staging_field in staging_row and published_field in published_row):
                staging_value = staging_row[staging_field]
                published_value = published_row[published_field]
                
                # Compare values (handle NaN/None)
                staging_val_str = str(staging_value) if pd.notna(staging_value) else ""
                published_val_str = str(published_value) if pd.notna(published_value) else ""
                
                if staging_val_str != published_val_str:
                    differences.append((staging_field, published_field, staging_value, published_value))
        
        if not differences:
            continue  # No differences for this model
        
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(common_models)}: {staging_name} ↔ {published_name}")
        print(f"{'='*80}")
        
        # Show current verification status
        current_verified = str(staging_row.get('verified', '')).lower() == 'y'
        verification_status = "✓ VERIFIED" if current_verified else "⚠ UNVERIFIED"
        print(f"Current status: {verification_status}")
        if not current_verified:
            print("📝 Note: Selecting 'p' or 's' will mark this model as verified (protected from automation overwrites)")
        
        print(f"Found {len(differences)} field differences")
        
        for diff_idx, (staging_field, published_field, staging_value, published_value) in enumerate(differences, 1):
            print(f"\n[DIFF {diff_idx}/{len(differences)}] Field: {staging_field} ↔ {published_field}")
            print(f"  Staging:   {staging_value}")
            print(f"  Published: {published_value}")
            
            while True:
                choice = input("Resolution (p/s/skip/quit): ").strip().lower()
                
                if choice == 'quit':
                    print(f"\nExiting diff resolution. Made {updates_made} updates.")
                    return staging_df, published_df
                
                elif choice == 'skip':
                    break
                
                elif choice == 'p':
                    # Keep published value, update staging
                    staging_df.loc[staging_df['model'] == staging_name, staging_field] = published_value
                    # Mark staging model as verified to protect from automation overwrites
                    staging_df.loc[staging_df['model'] == staging_name, 'verified'] = 'y'
                    
                    # Add audit trail comment
                    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
                    audit_comment = f"Field {staging_field} manually updated via diff resolution on {timestamp}"
                    
                    # Append to notes field
                    current_notes = staging_df.loc[staging_df['model'] == staging_name, 'notes'].iloc[0]
                    if current_notes and pd.notna(current_notes) and str(current_notes).strip():
                        new_notes = f"{current_notes}. {audit_comment}"
                    else:
                        new_notes = audit_comment
                    staging_df.loc[staging_df['model'] == staging_name, 'notes'] = new_notes
                    
                    print(f"  ✓ Updated staging {staging_field} to: {published_value}")
                    print(f"  ✓ Marked {staging_name} as verified (protected from automation overwrites)")
                    updates_made += 1
                    break
                
                elif choice == 's':
                    # Keep staging value, update published
                    published_df.loc[published_df['Model'] == published_name, published_field] = staging_value
                    # Mark staging model as verified to protect the source value
                    staging_df.loc[staging_df['model'] == staging_name, 'verified'] = 'y'
                    
                    # Add audit trail comment
                    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
                    audit_comment = f"Field {staging_field} confirmed via diff resolution on {timestamp}"
                    
                    # Append to notes field  
                    current_notes = staging_df.loc[staging_df['model'] == staging_name, 'notes'].iloc[0]
                    if current_notes and pd.notna(current_notes) and str(current_notes).strip():
                        new_notes = f"{current_notes}. {audit_comment}"
                    else:
                        new_notes = audit_comment
                    staging_df.loc[staging_df['model'] == staging_name, 'notes'] = new_notes
                    
                    print(f"  ✓ Updated published {published_field} to: {staging_value}")
                    print(f"  ✓ Marked {staging_name} as verified (protected from automation overwrites)")
                    updates_made += 1
                    break
                
                else:
                    print("Invalid choice. Use 'p', 's', 'skip', or 'quit'")
    
    print(f"\n{'='*80}")
    print(f"DIFF RESOLUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total updates made: {updates_made}")
    
    return staging_df, published_df


def preview_sync(
    models_to_sync: List[str],
    source_df: pd.DataFrame,
    mapping: Dict[str, str],
    direction: str
) -> None:
    """Preview sync changes without making modifications.
    
    Args:
        models_to_sync: List of models that would be synced
        source_df: Source DataFrame
        mapping: Name mapping dictionary
        direction: Sync direction
    """
    if not models_to_sync:
        print("No models would be synced.")
        return
    
    print(f"\nDRY RUN: {len(models_to_sync)} models would be synced ({direction})")
    print(f"{'='*60}")
    
    for i, model_name in enumerate(models_to_sync, 1):
        print(f"\n{i}. {model_name}")
        try:
            translated = display_translation_preview(model_name, source_df, mapping, direction)
            target_name = translated.get('Model' if direction == 'staging_to_published' else 'model')
            print(f"   Would create: {target_name}")
        except Exception as e:
            print(f"   Error: {e}")


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize staging and published model datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check-mappings                    Check for unmapped models
  %(prog)s --suggest-mappings                  Generate mapping suggestions
  %(prog)s --dry-run --direction staging-to-published  Preview sync changes
  %(prog)s --sync --direction staging-to-published     Interactive sync
  %(prog)s --diff                              Interactive field value comparison and resolution
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Data directory (default: data)'
    )
    
    parser.add_argument(
        '--mapping-file',
        type=Path,
        default=Path('configs/model_name_mapping.yaml'),
        help='Model name mapping file (default: configs/model_name_mapping.yaml)'
    )
    
    parser.add_argument(
        '--check-mappings',
        action='store_true',
        help='Check for unmapped models and report status'
    )
    
    parser.add_argument(
        '--suggest-mappings',
        action='store_true',
        help='Generate mapping suggestions for unmapped models'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview sync changes without making modifications'
    )
    
    parser.add_argument(
        '--sync',
        action='store_true',
        help='Perform interactive sync with manual approval'
    )
    
    parser.add_argument(
        '--direction',
        choices=['staging-to-published', 'published-to-staging', 'both'],
        default='staging-to-published',
        help='Sync direction (default: staging-to-published)'
    )
    
    parser.add_argument(
        '--diff',
        action='store_true',
        help='Interactive field value comparison and resolution between datasets'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.check_mappings, args.suggest_mappings, args.dry_run, args.sync, args.diff]):
        parser.error("Must specify one of: --check-mappings, --suggest-mappings, --dry-run, --sync, --diff")
    
    if (args.dry_run or args.sync) and not args.direction:
        parser.error("Must specify --direction when using --dry-run or --sync")
    
    try:
        # Load datasets
        staging_path = args.data_dir / "staging" / "above_1e25_flop_staging.csv"
        published_path = args.data_dir / "published" / "published_data.csv"
        
        if not staging_path.exists():
            print(f"Error: Staging file not found: {staging_path}")
            sys.exit(1)
        
        if not published_path.exists():
            print(f"Error: Published file not found: {published_path}")
            sys.exit(1)
        
        staging_df = pd.read_csv(staging_path)
        published_df = pd.read_csv(published_path)
        
        print(f"Loaded staging dataset: {len(staging_df)} models")
        print(f"Loaded published dataset: {len(published_df)} models")
        
        # Load mapping
        try:
            mapping = load_name_mapping(args.mapping_file)
            print(f"Loaded name mapping: {len(mapping)} entries")
        except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
            print(f"Error loading mapping file: {e}")
            sys.exit(1)
        
        # Execute requested operation
        if args.check_mappings:
            unmapped_staging, unmapped_published = detect_unmapped_models(
                staging_df, published_df, mapping
            )
            
            print(f"\n{'='*60}")
            print("MAPPING STATUS")
            print(f"{'='*60}")
            print(f"Mapped staging models: {len(mapping)}")
            print(f"Unmapped staging models: {len(unmapped_staging)}")
            print(f"Unmapped published models: {len(unmapped_published)}")
            
            if unmapped_staging:
                print(f"\nUnmapped staging models:")
                for model in sorted(unmapped_staging):
                    print(f"  - {model}")
            
            if unmapped_published:
                print(f"\nUnmapped published models:")
                for model in sorted(unmapped_published):
                    print(f"  - {model}")
            
            if unmapped_staging or unmapped_published:
                print(f"\nRun with --suggest-mappings to generate suggestions")
                sys.exit(1)
            else:
                print(f"\n✓ All models have mappings!")
        
        elif args.suggest_mappings:
            unmapped_staging, _ = detect_unmapped_models(
                staging_df, published_df, mapping
            )
            
            if not unmapped_staging:
                print("No unmapped staging models found.")
                return
            
            suggestions = suggest_mappings(unmapped_staging)
            interactive_mapping_approval(suggestions, staging_df, args.mapping_file)
        
        elif args.diff:
            # Interactive field value comparison and resolution
            print(f"\n{'='*60}")
            print("INTERACTIVE FIELD DIFF RESOLUTION")
            print(f"{'='*60}")
            
            updated_staging_df, updated_published_df = interactive_diff_resolution(
                staging_df, published_df, mapping
            )
            
            # Save updated datasets if changes were made
            if not updated_staging_df.equals(staging_df):
                updated_staging_df.to_csv(staging_path, index=False)
                print(f"\n✓ Updated staging dataset saved to: {staging_path}")
            
            if not updated_published_df.equals(published_df):
                updated_published_df.to_csv(published_path, index=False)
                print(f"✓ Updated published dataset saved to: {published_path}")
        
        elif args.dry_run or args.sync:
            # First check for unmapped models before any sync operation
            unmapped_staging, unmapped_published = detect_unmapped_models(
                staging_df, published_df, mapping
            )
            
            # Determine which unmapped models matter for this sync direction
            blocking_unmapped = []
            if args.direction in ['staging-to-published', 'both']:
                blocking_unmapped.extend(unmapped_staging)
            if args.direction in ['published-to-staging', 'both']:
                blocking_unmapped.extend(unmapped_published)
            
            if blocking_unmapped:
                print(f"\n{'='*60}")
                print("⚠️  UNMAPPED MODELS DETECTED")
                print(f"{'='*60}")
                
                if unmapped_staging and args.direction in ['staging-to-published', 'both']:
                    print(f"\n📋 Unmapped staging models ({len(unmapped_staging)}):")
                    for model in unmapped_staging[:10]:  # Show first 10
                        print(f"  - {model}")
                    if len(unmapped_staging) > 10:
                        print(f"  ... and {len(unmapped_staging) - 10} more")
                
                if unmapped_published and args.direction in ['published-to-staging', 'both']:
                    print(f"\n📋 Unmapped published models ({len(unmapped_published)}):")
                    for model in unmapped_published[:10]:  # Show first 10
                        print(f"  - {model}")
                    if len(unmapped_published) > 10:
                        print(f"  ... and {len(unmapped_published) - 10} more")
                
                print(f"\n{'='*60}")
                print("❌ Cannot proceed with sync: All models must be mapped.")
                print(f"{'='*60}")
                print("\nRecommended actions:")
                print("1. Run with --suggest-mappings to interactively add mappings")
                print("2. Manually add mappings to configs/model_name_mapping.yaml")
                print("\nAlternatively, you can proceed without proper mappings (NOT RECOMMENDED)")
                
                if args.sync:  # Only ask for override in actual sync, not dry-run
                    override = input("\n⚠️  Proceed anyway without proper mappings? (yes/no): ").strip().lower()
                    if override != 'yes':
                        print("\n✓ Sync cancelled. Please add the missing mappings first.")
                        return
                    else:
                        print("\n⚠️  WARNING: Proceeding without proper mappings may cause data issues!")
                else:
                    # For dry-run, just exit
                    print("\nDry-run cancelled. Please add the missing mappings first.")
                    return
            
            # Execute sync operations
            direction_map = {
                'staging-to-published': 'staging_to_published',
                'published-to-staging': 'published_to_staging'
            }
            
            if args.direction == 'both':
                directions = ['staging_to_published', 'published_to_staging']
            else:
                directions = [direction_map[args.direction]]
            
            for direction in directions:
                print(f"\n{'='*60}")
                print(f"PROCESSING: {direction.replace('_', ' → ')}")
                print(f"{'='*60}")
                
                # Determine source and target datasets
                if direction == 'staging_to_published':
                    source_df, target_df = staging_df, published_df
                    target_path = published_path
                else:
                    source_df, target_df = published_df, staging_df
                    target_path = staging_path
                
                # Find models to sync
                models_to_sync = find_models_to_sync(source_df, target_df, mapping, direction)
                
                if args.dry_run:
                    preview_sync(models_to_sync, source_df, mapping, direction)
                else:
                    # Interactive sync
                    approved_models, removed_models = interactive_sync_approval(
                        models_to_sync, source_df, target_df, mapping, direction
                    )
                    
                    # Handle model removals from staging
                    if removed_models and direction == 'staging_to_published':
                        staging_df_updated = staging_df[~staging_df['model'].isin(removed_models)]
                        staging_df_updated.to_csv(staging_path, index=False)
                        print(f"\n🗑️  Removed {len(removed_models)} models from staging:")
                        for model in removed_models:
                            print(f"  - {model}")
                        print(f"✓ Updated staging file: {staging_path}")
                        
                        # Update the local staging_df reference for potential subsequent operations
                        staging_df = staging_df_updated
                    
                    # Perform sync for approved models
                    if approved_models:
                        success = perform_sync(approved_models, target_df, target_path, direction)
                        if not success:
                            print(f"Sync failed for direction: {direction}")
                            sys.exit(1)
                    else:
                        print("No models approved for sync.")
                
                if len(directions) > 1:
                    print()  # Add spacing between directions
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()