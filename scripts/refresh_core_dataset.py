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
from typing import Dict, List, Optional, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.models import Model, ThresholdClassification
from epoch_tracker.storage import JSONStorage


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
        if model.threshold_classification == ThresholdClassification.HIGH_CONFIDENCE_ABOVE:
            above_candidates.append(model)
        elif model.threshold_classification == ThresholdClassification.NOT_SURE:
            above_candidates.append(model)  # Uncertain models go to above table for verification
        elif model.threshold_classification == ThresholdClassification.HIGH_CONFIDENCE_BELOW:
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
        if model.threshold_classification in [
            ThresholdClassification.HIGH_CONFIDENCE_ABOVE,
            ThresholdClassification.NOT_SURE
        ]:
            candidates.append(model)
    
    # Deduplicate candidates by name, keeping highest FLOP
    return deduplicate_models_by_name(candidates)


def format_alternative_methods(model: Model) -> str:
    """
    Format alternative estimation methods in human-readable format.
    
    Args:
        model: Model object with alternative estimates
        
    Returns:
        Human-readable string describing alternative methods
    """
    if not model.alternative_estimates:
        return ""
    
    alternatives = []
    for est in model.alternative_estimates:
        method_name = est.method.value.replace('_', ' ').title()
        flop_str = f"{est.flop:.2e}" if est.flop else "N/A"
        confidence_str = est.confidence.value.title()
        alternatives.append(f"{method_name}: {flop_str} ({confidence_str})")
    
    return "; ".join(alternatives)


def model_to_csv_row(model: Model, verified: str = "") -> Dict:
    """
    Convert a Model object to a CSV row dictionary.
    
    Args:
        model: Model object to convert
        verified: Verification status ('y' if manually verified, empty otherwise)
        
    Returns:
        Dictionary representing a CSV row
    """
    return {
        'model': model.name,  # Renamed from 'name' to 'model'
        'developer': model.developer,
        'release_date': model.release_date.isoformat() if model.release_date else None,
        'parameters': model.parameters,
        'training_flop': f"{model.training_flop:.2e}" if model.training_flop else None,
        'confidence': model.training_flop_confidence.value,
        'estimation_method': model.estimation_method.value,
        'alternative_methods': format_alternative_methods(model),
        'threshold_classification': model.threshold_classification.value,
        'status': model.status.value,
        'reasoning': model.reasoning,
        'sources': "; ".join(model.sources) if model.sources else "",
        'verified': verified,
        'last_updated': model.last_updated.isoformat(),
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
    
    # Drop the temporary numeric column
    df = df.drop('training_flop_numeric', axis=1)
    
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
    
    # Drop the temporary numeric column
    updated_df = updated_df.drop('training_flop_numeric', axis=1)
    
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
    
    # Detect duplicates and coverage issues
    duplicates = detect_duplicates(above_df, below_df)
    coverage = check_model_coverage(deduplicated_models, above_df, below_df)
    
    # Print comprehensive summary
    above_verified = len(above_df[above_df['verified'].fillna('').str.lower() == 'y'])
    below_verified = len(below_df[below_df['verified'].fillna('').str.lower() == 'y'])
    
    high_confidence_above = len(above_df[above_df['threshold_classification'] == 'high_confidence_above_1e25'])
    not_sure = len(above_df[(above_df['threshold_classification'] == 'not_sure') & (above_df['verified'].fillna('').str.lower() != 'y')])
    high_confidence_below = len(below_df[below_df['threshold_classification'] == 'high_confidence_below_1e25'])
    
    print(f"\n{'='*70}")
    print("DUAL-TABLE CORE DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Above/Uncertain Table (above_1e25_flop.csv):")
    print(f"  Total models: {len(above_df)}")
    print(f"  Manually verified: {above_verified}")
    print(f"  High confidence above 1e25: {high_confidence_above}")
    print(f"  Uncertain (need verification): {not_sure}")
    
    print(f"\nBelow Table (below_1e25_flop.csv):")
    print(f"  Total models: {len(below_df)}")
    print(f"  Manually verified: {below_verified}")
    print(f"  High confidence below 1e25: {high_confidence_below}")
    
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