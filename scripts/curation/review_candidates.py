#!/usr/bin/env python3
"""
Semi-automatic process for reviewing candidate models against the staging dataset.

This script compares models in clean/above_1e25_flop.csv against staging/above_1e25_flop_staging.csv
and allows interactive user review of new candidates.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_datasets(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load candidate and staging datasets.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Tuple of (df_candidates, df_staging)
    """
    candidates_path = data_dir / "clean" / "above_1e25_flop.csv"
    staging_path = data_dir / "staging" / "above_1e25_flop_staging.csv"
    
    if not candidates_path.exists():
        print(f"Error: Candidates file not found: {candidates_path}")
        sys.exit(1)
    
    if not staging_path.exists():
        print(f"Warning: Staging file not found: {staging_path}")
        print("Creating empty staging dataset...")
        # Create empty staging with same columns as candidates
        df_candidates = pd.read_csv(candidates_path)
        df_staging = pd.DataFrame(columns=df_candidates.columns)
        # Ensure staging directory exists
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        df_staging.to_csv(staging_path, index=False)
    else:
        df_staging = pd.read_csv(staging_path)
    
    df_candidates = pd.read_csv(candidates_path)
    
    return df_candidates, df_staging


def find_new_candidates(df_candidates: pd.DataFrame, df_staging: pd.DataFrame) -> pd.DataFrame:
    """Find models in candidates that don't exist in staging.
    
    Args:
        df_candidates: Candidate models dataframe
        df_staging: Staging models dataframe
        
    Returns:
        DataFrame with new candidate models
    """
    # Get unique model names from staging
    staging_models = set(df_staging['model'].str.lower()) if len(df_staging) > 0 else set()
    
    # Find candidates not in staging
    new_candidates = df_candidates[~df_candidates['model'].str.lower().isin(staging_models)]
    
    return new_candidates.copy()


def display_model_info(model_row: pd.Series) -> None:
    """Display all relevant fields for a model."""
    print("\n" + "="*80)
    print(f"MODEL: {model_row['model']}")
    print("="*80)
    
    # Key fields to display prominently
    key_fields = [
        ('Developer', 'developer'),
        ('Training FLOP', 'training_flop'),
        ('Confidence', 'confidence'),
        ('Parameters', 'parameters'),
        ('Estimation Method', 'estimation_method'),
        ('Threshold Classification', 'threshold_classification'),
        ('Status', 'status'),
    ]
    
    for label, field in key_fields:
        if field in model_row and pd.notna(model_row[field]):
            print(f"{label}: {model_row[field]}")
    
    # Display reasoning if available
    if 'reasoning' in model_row and pd.notna(model_row['reasoning']):
        print(f"\nReasoning: {model_row['reasoning']}")
    
    # Display sources if available
    if 'sources' in model_row and pd.notna(model_row['sources']):
        print(f"\nSources: {model_row['sources']}")
    
    # Display alternative methods if available
    if 'alternative_methods' in model_row and pd.notna(model_row['alternative_methods']):
        print(f"\nAlternative Methods: {model_row['alternative_methods']}")
    
    print("="*80)


def move_to_below_threshold(model_row: pd.Series, data_dir: Path, reason: str = "") -> bool:
    """Move a model from above to below threshold CSV with verified=y.
    
    Args:
        model_row: Model data to move
        data_dir: Base data directory
        reason: User-provided reason for the decision
        
    Returns:
        True if successful, False otherwise
    """
    above_path = data_dir / "clean" / "above_1e25_flop.csv"
    below_path = data_dir / "clean" / "below_1e25_flop.csv"
    
    try:
        # Load both CSVs
        df_above = pd.read_csv(above_path)
        
        if below_path.exists():
            df_below = pd.read_csv(below_path)
        else:
            # Create empty below dataset with same columns
            df_below = pd.DataFrame(columns=df_above.columns)
        
        # Remove from above dataset
        df_above = df_above[df_above['model'].str.lower() != model_row['model'].lower()]
        
        # Prepare row for below dataset
        new_row = model_row.copy()
        new_row['verified'] = 'y'
        new_row['last_updated'] = datetime.utcnow().isoformat()
        new_row['notes'] = f'Moved from above_1e25_flop.csv by manual review. Reason: {reason}' if reason else 'Moved from above_1e25_flop.csv by manual review'
        
        # Update status and threshold classification
        new_row['status'] = 'confirmed_below_1e25'
        if 'threshold_classification' in new_row:
            new_row['threshold_classification'] = 'high_confidence_below_1e25'
        
        # Add to below dataset
        df_below = pd.concat([df_below, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save both CSVs
        df_above.to_csv(above_path, index=False)
        df_below.to_csv(below_path, index=False)
        
        print(f"\nSuccessfully moved {model_row['model']} to below_1e25_flop.csv with verified=y")
        return True
        
    except Exception as e:
        print(f"\nError moving model: {e}")
        return False


def add_to_staging(model_row: pd.Series, data_dir: Path, reason: str = "") -> bool:
    """Add a model to the staging dataset and mark as verified in candidates.
    
    Args:
        model_row: Model data to add
        data_dir: Base data directory
        reason: User-provided reason for the decision
        
    Returns:
        True if successful, False otherwise
    """
    staging_path = data_dir / "staging" / "above_1e25_flop_staging.csv"
    above_path = data_dir / "clean" / "above_1e25_flop.csv"
    
    try:
        # Load staging dataset
        df_staging = pd.read_csv(staging_path)
        
        # Add timestamp and reason to staging
        staging_row = model_row.copy()
        staging_row['last_updated'] = datetime.utcnow().isoformat()
        staging_row['notes'] = f'Added to staging by manual review. Reason: {reason}' if reason else 'Added to staging by manual review'
        
        # Add to staging
        df_staging = pd.concat([df_staging, pd.DataFrame([staging_row])], ignore_index=True)
        
        # Save staging
        df_staging.to_csv(staging_path, index=False)
        
        # Mark as verified in original above_1e25_flop.csv
        df_above = pd.read_csv(above_path)
        mask = df_above['model'].str.lower() == model_row['model'].lower()
        if mask.any():
            df_above.loc[mask, 'verified'] = 'y'
            df_above.loc[mask, 'last_updated'] = datetime.utcnow().isoformat()
            if reason:
                current_notes = df_above.loc[mask, 'notes'].iloc[0] if pd.notna(df_above.loc[mask, 'notes'].iloc[0]) else ''
                new_notes = f'{current_notes}; Manual review - confirmed above 1e25 FLOP. Reason: {reason}'.strip('; ')
                df_above.loc[mask, 'notes'] = new_notes
            else:
                current_notes = df_above.loc[mask, 'notes'].iloc[0] if pd.notna(df_above.loc[mask, 'notes'].iloc[0]) else ''
                new_notes = f'{current_notes}; Manual review - confirmed above 1e25 FLOP'.strip('; ')
                df_above.loc[mask, 'notes'] = new_notes
            df_above.to_csv(above_path, index=False)
        
        print(f"\nSuccessfully added {model_row['model']} to staging dataset and marked as verified")
        return True
        
    except Exception as e:
        print(f"\nError adding to staging: {e}")
        return False


def main():
    """Main entry point for candidate review process."""
    parser = argparse.ArgumentParser(
        description="Review candidate models against staging dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of models to review before asking to continue (default: 10)"
    )
    
    args = parser.parse_args()
    
    print("Loading datasets...")
    df_candidates, df_staging = load_datasets(args.data_dir)
    
    print(f"Candidates: {len(df_candidates)} models")
    print(f"Staging: {len(df_staging)} models")
    
    # Find new candidates
    new_candidates = find_new_candidates(df_candidates, df_staging)
    
    if len(new_candidates) == 0:
        print("\nNo new candidates to review!")
        return
    
    print(f"\nFound {len(new_candidates)} new candidates to review")
    print("\nOptions for each model:")
    print("  y - Add to staging dataset (model is above 1e25 FLOP) - will prompt for reason")
    print("  n - Move to below_1e25_flop.csv with verified=y (model is below threshold) - will prompt for reason")
    print("  skip - Skip this model for now")
    print("  quit - Exit the review process")
    
    # Process each candidate
    reviewed = 0
    added_to_staging = 0
    moved_to_below = 0
    skipped = 0
    
    for idx, (_, model_row) in enumerate(new_candidates.iterrows()):
        # Display model information
        display_model_info(model_row)
        
        # Get user input
        while True:
            response = input("\nDecision (y/n/skip/quit): ").strip().lower()
            
            if response == 'quit':
                print("\nExiting review process...")
                break
            elif response == 'y':
                reason = input("Please provide a reason for confirming this model is above 1e25 FLOP: ").strip()
                if add_to_staging(model_row, args.data_dir, reason):
                    added_to_staging += 1
                break
            elif response == 'n':
                reason = input("Please provide a reason for confirming this model is below 1e25 FLOP: ").strip()
                if move_to_below_threshold(model_row, args.data_dir, reason):
                    moved_to_below += 1
                break
            elif response == 'skip':
                print(f"Skipping {model_row['model']}")
                skipped += 1
                break
            else:
                print("Invalid response. Please enter 'y', 'n', 'skip', or 'quit'")
        
        reviewed += 1
        
        if response == 'quit':
            break
        
        # Check if we should ask to continue
        if reviewed % args.batch_size == 0 and idx < len(new_candidates) - 1:
            cont = input(f"\nReviewed {reviewed} models. Continue? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    # Print summary
    print("\n" + "="*60)
    print("REVIEW SUMMARY")
    print("="*60)
    print(f"Total reviewed: {reviewed}")
    print(f"Added to staging: {added_to_staging}")
    print(f"Moved to below threshold: {moved_to_below}")
    print(f"Skipped: {skipped}")
    print(f"Remaining candidates: {len(new_candidates) - reviewed}")
    print("="*60)


if __name__ == "__main__":
    main()