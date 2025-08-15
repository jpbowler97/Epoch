#!/usr/bin/env python3
"""
Refresh core datasets for models above and below 1e25 FLOP threshold.

This script manages two curated CSV files:
- data/clean/above_1e25_flop.csv: Models that are candidates for being above 1e25 FLOP
- data/clean/below_1e25_flop.csv: Models that are confidently below 1e25 FLOP

The system ensures that:
1. Every model from estimated_models.json appears in exactly one CSV file
2. Models can move between CSVs based on updated FLOP estimates
3. Models with 'verified=y' are protected from automatic updates
4. Duplicates across both tables are detected and flagged
"""

import argparse
import logging
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.models import Model
from epoch_tracker.storage import JSONStorage
from epoch_tracker.config.thresholds import get_threshold_config, ThresholdClassification, CANDIDATE_CLASSIFICATIONS


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def deduplicate_models_by_name(models: List[Model]) -> List[Model]:
    """
    Deduplicate models by name, keeping the one with highest training FLOP.
    
    For models with same name, preference order:
    1. Highest training_flop value
    2. If same FLOP, highest confidence level
    3. If same confidence, most recent last_updated
    
    Args:
        models: List of models potentially containing duplicates
        
    Returns:
        List of deduplicated models
    """
    seen = {}
    confidence_order = {
        'high': 4,
        'medium': 3,
        'low': 2,
        'speculative': 1
    }
    
    for model in models:
        key = model.name.lower().strip()
        
        if key not in seen:
            seen[key] = model
        else:
            existing = seen[key]
            
            # Compare training FLOP (prefer higher)
            model_flop = model.training_flop or 0
            existing_flop = existing.training_flop or 0
            
            if model_flop > existing_flop:
                seen[key] = model
            elif model_flop == existing_flop:
                # Same FLOP, compare confidence
                model_conf = confidence_order.get(model.training_flop_confidence.value, 0)
                existing_conf = confidence_order.get(existing.training_flop_confidence.value, 0)
                
                if model_conf > existing_conf:
                    seen[key] = model
                elif model_conf == existing_conf:
                    # Same confidence, prefer more recent
                    if model.last_updated > existing.last_updated:
                        seen[key] = model
    
    deduplicated = list(seen.values())
    original_count = len(models)
    final_count = len(deduplicated)
    
    if original_count != final_count:
        logging.info(f"Deduplicated {original_count} models to {final_count} unique models")
    
    return deduplicated


def categorize_models(models: List[Model]) -> Dict[str, List[Model]]:
    """
    Categorize models into above/below threshold groups.
    
    Categories:
    - above_candidates: HIGH_CONFIDENCE_ABOVE + NOT_SURE (need manual verification)
    - below_confident: HIGH_CONFIDENCE_BELOW (confidently below threshold)
    
    Args:
        models: List of all models (deduplicated)
        
    Returns:
        Dict with 'above_candidates' and 'below_confident' model lists
    """
    above_candidates = []
    below_confident = []
    
    for model in models:
        # Use status field instead of threshold_classification for model placement
        if model.status in [ModelStatus.CONFIRMED_ABOVE, ModelStatus.LIKELY_ABOVE]:
            above_candidates.append(model)
        elif model.status == ModelStatus.UNCERTAIN:
            above_candidates.append(model)  # Uncertain models go to above table for verification
        else:  # CONFIRMED_BELOW, LIKELY_BELOW
            below_confident.append(model)
    
    return {
        'above_candidates': above_candidates,
        'below_confident': below_confident
    }


def get_candidate_models(models: List[Model]) -> List[Model]:
    """
    Filter models to get candidates for above 1e25 FLOP.
    
    Includes:
    - HIGH_CONFIDENCE_ABOVE: Models we're confident are above threshold
    - NOT_SURE: Models in the uncertain range that need manual verification
    
    Excludes:
    - HIGH_CONFIDENCE_BELOW: Models we're confident are below threshold
    
    Also deduplicates models by name, keeping the highest FLOP estimate.
    
    Args:
        models: List of all models
        
    Returns:
        List of candidate models (deduplicated)
    """
    candidates = []
    
    for model in models:
        # Filter based on threshold classification instead of status
        threshold_class = model.get_threshold_classification()
        if threshold_class in CANDIDATE_CLASSIFICATIONS:
            candidates.append(model)
    
    # Deduplicate candidates by name, keeping highest FLOP
    return deduplicate_models_by_name(candidates)


def format_alternative_methods(model: Model) -> str:
    """
    Format alternative estimation methods in human-readable format.
    Includes original uncapped estimate for blacklist-capped models.
    
    Args:
        model: Model object with alternative estimates
        
    Returns:
        Human-readable string describing alternative methods
    """
    alternatives = []
    
    # Add regular alternative estimates
    if model.alternative_estimates:
        for est in model.alternative_estimates:
            method_name = est.method.value.replace('_', ' ').title()
            flop_str = f"{est.flop:.2e}" if est.flop else "N/A"
            confidence_str = est.confidence.value.title()
            alternatives.append(f"{method_name}: {flop_str} ({confidence_str})")
    
    # Add original estimate for blacklist-capped models
    if is_blacklist_capped(model):
        original_estimate = get_original_estimate(model)
        if original_estimate:
            alternatives.append(f"Original (uncapped): {original_estimate} FLOP")
    
    return "; ".join(alternatives)


def generate_confidence_explanation(model: Model) -> str:
    """
    Generate human-readable explanation for confidence level.
    
    Args:
        model: Model object with confidence and estimation method
        
    Returns:
        Human-readable explanation of confidence level
    """
    confidence = model.training_flop_confidence.value
    method = model.estimation_method.value
    
    explanations = {
        'high': {
            'epoch_estimate': 'Official disclosure or high-precision research estimate',
            'known_specification': 'Published model specifications and training details',
            'scaling_laws': 'Known parameters with documented training tokens',
            'benchmark_based': 'Very close benchmark match to reference model'
        },
        'medium': {
            'epoch_estimate': 'Reliable research estimate from industry analysis',
            'known_specification': 'Industry estimates of specifications',
            'scaling_laws': 'Known parameters with estimated training tokens',
            'benchmark_based': 'Good benchmark match with multiple agreeing sources'
        },
        'low': {
            'epoch_estimate': 'Speculative research estimate',
            'known_specification': 'Estimated specifications from limited data',
            'scaling_laws': 'Extracted parameters with uncertain training tokens',
            'benchmark_based': 'Distant benchmark interpolation or single source'
        },
        'speculative': {
            'epoch_estimate': 'Highly speculative estimate',
            'known_specification': 'Rough estimates from minimal information',
            'scaling_laws': 'Heuristic parameter extraction with assumed ratios',
            'benchmark_based': 'Very distant benchmark extrapolation'
        }
    }
    
    return explanations.get(confidence, {}).get(method, f'{confidence.title()} confidence from {method.replace("_", " ")} method')


def is_blacklist_capped(model: Model) -> bool:
    """Check if a model was capped due to developer blacklist policy."""
    if not model.reasoning:
        return False
    return "capped" in model.reasoning.lower() and "developer blacklist policy" in model.reasoning.lower()


def get_original_estimate(model: Model) -> Optional[str]:
    """Extract original FLOP estimate from alternative estimates if model was capped."""
    if not is_blacklist_capped(model):
        return None
    
    # Look for original estimate in alternative estimates
    for alt_est in model.alternative_estimates:
        if "original estimate before developer policy cap" in alt_est.reasoning.lower():
            return f"{alt_est.flop:.2e}"
    
    # Try to parse from reasoning field as fallback
    reasoning = model.reasoning.lower()
    if "original estimate:" in reasoning:
        try:
            # Extract value like "Original estimate: 3.2e25 FLOP"
            parts = reasoning.split("original estimate:")
            if len(parts) > 1:
                value_part = parts[1].strip().split()[0]  # Get first part before space
                return value_part
        except:
            pass
    
    return None


def model_to_csv_row(model: Model, verified: str = "") -> Dict:
    """
    Convert a Model object to a CSV row dictionary.
    
    Args:
        model: Model object to convert
        verified: Verification status ('y' if manually verified, empty otherwise)
        
    Returns:
        Dictionary representing a CSV row
    """
    # Check if model was blacklist capped
    is_capped = is_blacklist_capped(model)
    
    return {
        'model': model.name,  # Renamed from 'name' to 'model'
        'developer': model.developer,
        'release_date': model.release_date.isoformat() if model.release_date else None,
        'parameters': model.parameters,
        'parameter_source': model.parameter_source,
        'training_flop': f"{model.training_flop:.2e}" if model.training_flop else None,
        'confidence': model.training_flop_confidence.value,
        'confidence_explanation': generate_confidence_explanation(model),
        'estimation_method': model.estimation_method.value,
        'alternative_methods': format_alternative_methods(model),
        'threshold_classification': model.get_threshold_classification(),
        'reasoning': model.reasoning,
        'sources': "; ".join(model.sources) if model.sources else "",
        'verified': verified,
        'last_updated': model.last_updated.isoformat(),
        'blacklist_status': 'capped' if is_capped else 'allowed',
        'notes': ""  # Empty field for manual notes
    }


def load_existing_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load existing CSV file if it exists.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame if file exists, None otherwise
    """
    if not csv_path.exists():
        return None
        
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded existing CSV with {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load existing CSV from {csv_path}: {e}")
        return None


def create_new_csv(models: List[Model], csv_path: Path) -> pd.DataFrame:
    """
    Create a new CSV file from candidate models.
    
    Args:
        models: List of candidate models
        csv_path: Path where CSV should be saved
        
    Returns:
        DataFrame containing the new data
    """
    logging.info(f"Creating new CSV with {len(models)} candidate models")
    
    rows = []
    for model in models:
        rows.append(model_to_csv_row(model))
    
    df = pd.DataFrame(rows)
    
    # Convert training_flop to numeric for proper sorting
    df['training_flop_numeric'] = pd.to_numeric(df['training_flop'], errors='coerce')
    
    # Sort by training FLOP (descending) then by model name
    df = df.sort_values(['training_flop_numeric', 'model'], ascending=[False, True], na_position='last')
    
    # Drop the temporary numeric column and any unwanted columns
    df = df.drop('training_flop_numeric', axis=1)
    
    # Remove unwanted columns that might have been preserved from old CSV files
    unwanted_columns = ['training_flop_formatted']
    for col in unwanted_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Ensure directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    logging.info(f"Created new CSV file: {csv_path}")
    
    return df


def detect_duplicates(above_df: pd.DataFrame, below_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect duplicate models across both CSV files.
    
    Args:
        above_df: DataFrame for above threshold models
        below_df: DataFrame for below threshold models
        
    Returns:
        Dict with duplicate information
    """
    duplicates = {
        'cross_table': [],  # Models in both tables
        'within_above': [],  # Duplicates within above table
        'within_below': []   # Duplicates within below table
    }
    
    # Check for cross-table duplicates
    above_models = set(above_df['model'].str.lower())
    below_models = set(below_df['model'].str.lower())
    cross_duplicates = above_models.intersection(below_models)
    
    for model_name in cross_duplicates:
        duplicates['cross_table'].append(model_name)
    
    # Check for within-table duplicates
    above_counts = above_df['model'].str.lower().value_counts()
    within_above_dups = above_counts[above_counts > 1].index.tolist()
    duplicates['within_above'] = within_above_dups
    
    below_counts = below_df['model'].str.lower().value_counts()
    within_below_dups = below_counts[below_counts > 1].index.tolist()
    duplicates['within_below'] = within_below_dups
    
    return duplicates


def deduplicate_within_table(df: pd.DataFrame, table_name: str, model_lookup: Dict[str, Model]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove duplicate models within a single table, keeping the one with highest FLOP estimate.
    
    Args:
        df: DataFrame to deduplicate
        table_name: Name of table for logging ("above" or "below")
        model_lookup: Dict mapping model names to Model objects for FLOP lookup
        
    Returns:
        Tuple of (deduplicated_df, resolution_log)
    """
    # Find duplicate model names (case-insensitive)
    df_lower = df.copy()
    df_lower['model_lower'] = df['model'].str.lower()
    
    duplicate_counts = df_lower['model_lower'].value_counts()
    duplicate_names = duplicate_counts[duplicate_counts > 1].index.tolist()
    
    if not duplicate_names:
        return df, []
    
    resolution_log = []
    indices_to_remove = []
    
    for model_name_lower in duplicate_names:
        # Get all rows with this model name
        duplicate_rows = df_lower[df_lower['model_lower'] == model_name_lower]
        
        if len(duplicate_rows) <= 1:
            continue
            
        # Determine which row to keep (highest FLOP estimate)
        best_row_idx = None
        best_flop = -1
        actual_model_name = duplicate_rows.iloc[0]['model']  # Use first occurrence's casing
        
        for idx, row in duplicate_rows.iterrows():
            # Try to get FLOP from model lookup first
            model_obj = model_lookup.get(model_name_lower)
            if model_obj and model_obj.training_flop:
                flop = model_obj.training_flop
            else:
                # Fall back to CSV training_flop field
                flop_str = str(row.get('training_flop', '')).strip()
                try:
                    flop = float(flop_str) if flop_str and flop_str != 'nan' else 0
                except (ValueError, TypeError):
                    flop = 0
            
            if flop > best_flop:
                best_flop = flop
                best_row_idx = idx
        
        # Mark all other rows for removal
        for idx, row in duplicate_rows.iterrows():
            if idx != best_row_idx:
                indices_to_remove.append(idx)
        
        # Log the resolution
        if best_flop > 0:
            resolution_log.append(f"✓ {actual_model_name}: Deduplicated in {table_name} table, kept entry with {best_flop:.2e} FLOP")
        else:
            resolution_log.append(f"✓ {actual_model_name}: Deduplicated in {table_name} table, kept first entry (no FLOP data)")
    
    # Remove duplicate rows
    if indices_to_remove:
        df_deduplicated = df.drop(index=indices_to_remove).reset_index(drop=True)
    else:
        df_deduplicated = df
    
    return df_deduplicated, resolution_log


def resolve_duplicates(above_df: pd.DataFrame, below_df: pd.DataFrame, all_models: List[Model]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Resolve cross-table duplicates by re-applying categorization logic.
    
    For models that appear in both tables:
    1. Check verified=y flag first - preserve manually verified models
    2. For unverified duplicates, re-apply threshold_classification logic
    3. Remove model from incorrect table
    4. Log resolution decisions
    
    Args:
        above_df: DataFrame for above threshold models
        below_df: DataFrame for below threshold models  
        all_models: All models from estimated_models.json for FLOP lookup
        
    Returns:
        Tuple of (updated_above_df, updated_below_df, resolution_log)
    """
    # Load centralized threshold configuration
    threshold_config = get_threshold_config()
    
    # Create model lookup for FLOP estimates
    model_lookup = {model.name.lower(): model for model in all_models}
    
    # Find cross-table duplicates
    above_models = set(above_df['model'].str.lower())
    below_models = set(below_df['model'].str.lower())
    cross_duplicates = above_models.intersection(below_models)
    
    if not cross_duplicates:
        return above_df, below_df, []
    
    resolution_log = []
    models_to_remove_from_above = []
    models_to_remove_from_below = []
    
    for model_name_lower in cross_duplicates:
        # Get actual model name (preserving case)
        above_match = above_df[above_df['model'].str.lower() == model_name_lower]
        below_match = below_df[below_df['model'].str.lower() == model_name_lower]
        
        if len(above_match) == 0 or len(below_match) == 0:
            continue
            
        above_row = above_match.iloc[0]
        below_row = below_match.iloc[0]
        actual_model_name = above_row['model']  # Use above table's casing
        
        # Check if either is manually verified
        above_verified = str(above_row.get('verified', '')).lower() == 'y'
        below_verified = str(below_row.get('verified', '')).lower() == 'y'
        
        if above_verified and below_verified:
            # Both verified - keep both and log warning
            resolution_log.append(f"⚠️  {actual_model_name}: Both tables have verified=y, keeping both (manual review needed)")
            continue
        elif above_verified:
            # Above is verified, remove from below
            models_to_remove_from_below.append(model_name_lower)
            resolution_log.append(f"✓ {actual_model_name}: Kept in above table (verified=y), removed from below")
            continue
        elif below_verified:
            # Below is verified, remove from above  
            models_to_remove_from_above.append(model_name_lower)
            resolution_log.append(f"✓ {actual_model_name}: Kept in below table (verified=y), removed from above")
            continue
        
        # Neither verified - use FLOP-based categorization
        model_obj = model_lookup.get(model_name_lower)
        if not model_obj or not model_obj.training_flop:
            # No FLOP data - keep in above table for manual review
            models_to_remove_from_below.append(model_name_lower)
            resolution_log.append(f"✓ {actual_model_name}: No FLOP data, kept in above table for review")
            continue
        
        # Apply threshold classification logic
        flop = model_obj.training_flop
        if flop >= threshold_config.high_confidence_above_threshold:  # HIGH_CONFIDENCE_ABOVE_THRESHOLD
            # Should be in above table
            models_to_remove_from_below.append(model_name_lower)
            resolution_log.append(f"✓ {actual_model_name}: {flop:.2e} FLOP ≥ {threshold_config.high_confidence_above_threshold:.1e}, moved to above table")
        elif flop <= threshold_config.high_confidence_below_threshold:  # HIGH_CONFIDENCE_BELOW_THRESHOLD  
            # Should be in below table
            models_to_remove_from_above.append(model_name_lower)
            resolution_log.append(f"✓ {actual_model_name}: {flop:.2e} FLOP ≤ {threshold_config.high_confidence_below_threshold:.1e}, moved to below table")
        else:
            # NOT_SURE range - should be in above table for verification
            models_to_remove_from_below.append(model_name_lower)
            resolution_log.append(f"✓ {actual_model_name}: {flop:.2e} FLOP in uncertain range, moved to above table for verification")
    
    # Remove duplicates from tables
    if models_to_remove_from_above:
        above_df = above_df[~above_df['model'].str.lower().isin(models_to_remove_from_above)]
    if models_to_remove_from_below:
        below_df = below_df[~below_df['model'].str.lower().isin(models_to_remove_from_below)]
    
    # Deduplicate within each table, keeping highest FLOP estimate
    above_df, above_dedup_log = deduplicate_within_table(above_df, "above", model_lookup)
    below_df, below_dedup_log = deduplicate_within_table(below_df, "below", model_lookup)
    
    # Combine all resolution logs
    all_logs = resolution_log + above_dedup_log + below_dedup_log
    
    return above_df, below_df, all_logs


def check_model_coverage(all_models: List[Model], above_df: pd.DataFrame, below_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Check that all models from estimated_models.json are covered by exactly one CSV.
    
    Args:
        all_models: All models from estimated_models.json
        above_df: DataFrame for above threshold models  
        below_df: DataFrame for below threshold models
        
    Returns:
        Dict with coverage information
    """
    # Get all model names from estimated_models.json
    all_model_names = set(model.name.lower() for model in all_models)
    
    # Get model names from both CSV files
    above_models = set(above_df['model'].str.lower())
    below_models = set(below_df['model'].str.lower())
    csv_models = above_models.union(below_models)
    
    # Find missing and extra models
    missing_from_csv = all_model_names - csv_models
    extra_in_csv = csv_models - all_model_names
    
    return {
        'missing_from_csv': list(missing_from_csv),
        'extra_in_csv': list(extra_in_csv),
        'total_estimated': len(all_model_names),
        'total_in_csv': len(csv_models)
    }


def manage_model_migrations(existing_above_df: pd.DataFrame, existing_below_df: pd.DataFrame, 
                          new_above_models: List[Model], new_below_models: List[Model]) -> Dict[str, List[str]]:
    """
    Identify models that need to move between tables based on updated FLOP estimates.
    
    Args:
        existing_above_df: Current above threshold CSV
        existing_below_df: Current below threshold CSV
        new_above_models: Updated models for above table
        new_below_models: Updated models for below table
        
    Returns:
        Dict with migration information
    """
    # Get current model locations
    current_above = set(existing_above_df['model'].str.lower())
    current_below = set(existing_below_df['model'].str.lower())
    
    # Get new model locations
    new_above = set(model.name.lower() for model in new_above_models)
    new_below = set(model.name.lower() for model in new_below_models)
    
    # Find migrations
    above_to_below = current_above.intersection(new_below)
    below_to_above = current_below.intersection(new_above)
    
    return {
        'above_to_below': list(above_to_below),
        'below_to_above': list(below_to_above)
    }


def refresh_existing_csv(existing_df: pd.DataFrame, models: List[Model], csv_path: Path) -> pd.DataFrame:
    """
    Refresh existing CSV with updated model data.
    
    Only updates rows where 'verified' is not 'y' and matching model data exists.
    Adds new models that aren't in the existing CSV.
    
    Args:
        existing_df: Existing DataFrame
        models: List of current candidate models
        csv_path: Path to save updated CSV
        
    Returns:
        Updated DataFrame
    """
    logging.info(f"Refreshing existing CSV with {len(existing_df)} existing rows")
    
    # Create lookup for existing models by (name, developer)
    model_lookup = {}
    for model in models:
        key = (model.name.lower(), model.developer.lower())
        model_lookup[key] = model
    
    # Track statistics
    updated_count = 0
    added_count = 0
    verified_count = 0
    
    # Process existing rows
    updated_rows = []
    processed_models = set()
    
    for _, row in existing_df.iterrows():
        # Handle both old 'name' field and new 'model' field for backward compatibility
        model_name = row.get('model', row.get('name', ''))
        key = (model_name.lower(), row['developer'].lower())
        processed_models.add(key)
        
        # Check if this row is verified
        if str(row.get('verified', '')).lower() == 'y':
            verified_count += 1
            # Update the row to use new field name if needed
            row_dict = row.to_dict()
            if 'name' in row_dict and 'model' not in row_dict:
                row_dict['model'] = row_dict.pop('name')
            updated_rows.append(row_dict)
            logging.debug(f"Keeping verified entry: {model_name}")
            continue
        
        # Try to find updated model data
        if key in model_lookup:
            updated_model = model_lookup[key]
            updated_row = model_to_csv_row(
                updated_model, 
                verified=row.get('verified', '')
            )
            # Preserve manual notes
            updated_row['notes'] = row.get('notes', '')
            updated_rows.append(updated_row)
            updated_count += 1
            logging.debug(f"Updated: {updated_model.name}")
        else:
            # Model no longer qualifies as candidate, but keep if not verified
            row_dict = row.to_dict()
            if 'name' in row_dict and 'model' not in row_dict:
                row_dict['model'] = row_dict.pop('name')
            updated_rows.append(row_dict)
            logging.debug(f"Keeping non-candidate (not verified): {model_name}")
    
    # Add new models that weren't in the existing CSV
    for model in models:
        key = (model.name.lower(), model.developer.lower())
        if key not in processed_models:
            updated_rows.append(model_to_csv_row(model))
            added_count += 1
            logging.debug(f"Added new candidate: {model.name}")
    
    # Create updated DataFrame
    updated_df = pd.DataFrame(updated_rows)
    
    # Convert training_flop to numeric for proper sorting
    updated_df['training_flop_numeric'] = pd.to_numeric(updated_df['training_flop'], errors='coerce')
    
    # Sort by training FLOP (descending) then by model name
    updated_df = updated_df.sort_values(['training_flop_numeric', 'model'], ascending=[False, True], na_position='last')
    
    # Drop the temporary numeric column and any unwanted columns
    updated_df = updated_df.drop('training_flop_numeric', axis=1)
    
    # Remove unwanted columns that might have been preserved from old CSV files
    unwanted_columns = ['training_flop_formatted']
    for col in unwanted_columns:
        if col in updated_df.columns:
            updated_df = updated_df.drop(col, axis=1)
    
    # Save updated CSV
    updated_df.to_csv(csv_path, index=False)
    
    logging.info(f"Refresh complete:")
    logging.info(f"  - Total rows: {len(updated_df)}")
    logging.info(f"  - Updated: {updated_count}")
    logging.info(f"  - Added: {added_count}")
    logging.info(f"  - Verified (unchanged): {verified_count}")
    logging.info(f"  - Saved to: {csv_path}")
    
    return updated_df


def main():
    """Main entry point for dual-table core dataset refresh."""
    parser = argparse.ArgumentParser(
        description="Refresh core datasets for models above and below 1e25 FLOP threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Create or refresh both datasets
  %(prog)s --reset            # Reset both datasets from scratch
  %(prog)s --verbose          # Enable detailed logging
  
The script manages two CSV files:
- data/clean/above_1e25_flop.csv: Models above or uncertain about 1e25 FLOP
- data/clean/below_1e25_flop.csv: Models confidently below 1e25 FLOP

Models with 'verified=y' are preserved during refresh operations.
"""
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset both datasets completely from scratch"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting dual-table core dataset refresh...")
    
    # Set up paths
    clean_dir = args.data_dir / "clean"
    above_csv_path = clean_dir / "above_1e25_flop.csv"
    below_csv_path = clean_dir / "below_1e25_flop.csv"
    
    # Initialize storage and load estimated models
    storage = JSONStorage(args.data_dir)
    
    try:
        collection = storage.load_models("estimated_models", stage="estimated")
        if not collection:
            logger.error("Could not load estimated_models.json. Run estimate_flops.py first.")
            return 1
            
        all_models = collection.models
        logger.info(f"Loaded {len(all_models)} models from estimated_models.json")
        
    except Exception as e:
        logger.error(f"Failed to load estimated models: {e}")
        return 1
    
    # Deduplicate and categorize models
    deduplicated_models = deduplicate_models_by_name(all_models)
    categorized = categorize_models(deduplicated_models)
    
    above_models = categorized['above_candidates']
    below_models = categorized['below_confident']
    
    logger.info(f"Categorized models: {len(above_models)} above/uncertain, {len(below_models)} below")
    
    # Load existing CSV files
    existing_above_df = load_existing_csv(above_csv_path) if above_csv_path.exists() else pd.DataFrame()
    existing_below_df = load_existing_csv(below_csv_path) if below_csv_path.exists() else pd.DataFrame()
    
    # Check for model migrations if not resetting
    if not args.reset and not existing_above_df.empty and not existing_below_df.empty:
        migrations = manage_model_migrations(existing_above_df, existing_below_df, above_models, below_models)
        if migrations['above_to_below'] or migrations['below_to_above']:
            logger.info("Model migrations detected:")
            for model in migrations['above_to_below']:
                logger.info(f"  Moving to below table: {model}")
            for model in migrations['below_to_above']:
                logger.info(f"  Moving to above table: {model}")
    
    # Create or refresh CSV files
    if args.reset or above_csv_path.exists() == False:
        logger.info("Creating/resetting above_1e25_flop.csv")
        above_df = create_new_csv(above_models, above_csv_path)
    else:
        logger.info("Refreshing above_1e25_flop.csv")
        above_df = refresh_existing_csv(existing_above_df, above_models, above_csv_path)
    
    if args.reset or below_csv_path.exists() == False:
        logger.info("Creating/resetting below_1e25_flop.csv")
        below_df = create_new_csv(below_models, below_csv_path)
    else:
        logger.info("Refreshing below_1e25_flop.csv")
        below_df = refresh_existing_csv(existing_below_df, below_models, below_csv_path)
    
    # Resolve cross-table duplicates automatically  
    above_df, below_df, resolution_log = resolve_duplicates(above_df, below_df, deduplicated_models)
    
    # Save resolved CSVs if any changes were made
    if resolution_log:
        above_df.to_csv(above_csv_path, index=False)
        below_df.to_csv(below_csv_path, index=False)
        print(f"\n🔧 DUPLICATE RESOLUTION ({len(resolution_log)} conflicts resolved):")
        for log_entry in resolution_log:
            print(f"  {log_entry}")
        print(f"\n✅ Updated CSV files saved automatically")
    
    # Detect duplicates and coverage issues
    duplicates = detect_duplicates(above_df, below_df)
    coverage = check_model_coverage(deduplicated_models, above_df, below_df)
    
    # Print comprehensive summary
    above_verified = len(above_df[above_df['verified'].fillna('').str.lower() == 'y'])
    below_verified = len(below_df[below_df['verified'].fillna('').str.lower() == 'y'])
    
    confirmed_above = len(above_df[above_df['status'] == 'confirmed_above_1e25'])
    likely_above = len(above_df[above_df['status'] == 'likely_above_1e25'])
    uncertain_unverified = len(above_df[(above_df['status'] == 'uncertain') & (above_df['verified'].fillna('').str.lower() != 'y')])
    confirmed_below = len(below_df[below_df['status'] == 'confirmed_below_1e25'])
    likely_below = len(below_df[below_df['status'] == 'likely_below_1e25'])
    
    print(f"\n{'='*70}")
    print("DUAL-TABLE CORE DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Above/Uncertain Table (above_1e25_flop.csv):")
    print(f"  Total models: {len(above_df)}")
    print(f"  Manually verified: {above_verified}")
    print(f"  Confirmed above 1e25: {confirmed_above}")
    print(f"  Likely above 1e25: {likely_above}")
    print(f"  Uncertain (need verification): {uncertain_unverified}")
    
    print(f"\nBelow Table (below_1e25_flop.csv):")
    print(f"  Total models: {len(below_df)}")
    print(f"  Manually verified: {below_verified}")
    print(f"  Confirmed below 1e25: {confirmed_below}")
    print(f"  Likely below 1e25: {likely_below}")
    
    print(f"\nCoverage Analysis:")
    print(f"  Models in estimated_models.json: {coverage['total_estimated']}")
    print(f"  Models in CSV files: {coverage['total_in_csv']}")
    print(f"  Missing from CSV: {len(coverage['missing_from_csv'])}")
    print(f"  Extra in CSV: {len(coverage['extra_in_csv'])}")
    
    # Report issues
    issues_found = False
    
    if duplicates['cross_table']:
        issues_found = True
        print(f"\n⚠️  CROSS-TABLE DUPLICATES ({len(duplicates['cross_table'])}):")
        for model in duplicates['cross_table'][:5]:  # Show first 5
            print(f"    - {model}")
        if len(duplicates['cross_table']) > 5:
            print(f"    ... and {len(duplicates['cross_table']) - 5} more")
    
    if duplicates['within_above']:
        issues_found = True
        print(f"\n⚠️  DUPLICATES IN ABOVE TABLE ({len(duplicates['within_above'])}):")
        for model in duplicates['within_above'][:3]:
            print(f"    - {model}")
    
    if duplicates['within_below']:
        issues_found = True
        print(f"\n⚠️  DUPLICATES IN BELOW TABLE ({len(duplicates['within_below'])}):")
        for model in duplicates['within_below'][:3]:
            print(f"    - {model}")
    
    if coverage['missing_from_csv']:
        issues_found = True
        print(f"\n⚠️  MODELS MISSING FROM CSV ({len(coverage['missing_from_csv'])}):")
        for model in coverage['missing_from_csv'][:5]:
            print(f"    - {model}")
        if len(coverage['missing_from_csv']) > 5:
            print(f"    ... and {len(coverage['missing_from_csv']) - 5} more")
    
    if not issues_found:
        print("\n✅ No duplicates or coverage issues detected!")
    
    print(f"\nDatasets saved to:")
    print(f"  - {above_csv_path}")
    print(f"  - {below_csv_path}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())