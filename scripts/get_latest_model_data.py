#!/usr/bin/env python3
"""
Unified data refresh script for all scrapers.

This script runs all available scrapers and prepares data for querying.
It serves as the main entry point for data collection in the Epoch tracker.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.scrapers import LMArenaScraper, OpenLMArenaWebScraper
from epoch_tracker.storage import JSONStorage
from epoch_tracker.utils.model_names import normalize_model_name


# =============================================================================
# MODEL EXCLUSION LIST
# =============================================================================
# 
# This list contains model names that should be excluded from data collection.
# These are typically generic, ambiguous, or low-quality model names that
# don't provide useful information for FLOP estimation.
#
# To add exclusions: Add model names to this list (case-insensitive matching)
# To remove exclusions: Remove names from this list
#
# Examples of models to exclude:
# - Generic names like "gpt", "claude", "llama" without version numbers
# - Test models, demos, or placeholder entries
# - Duplicates with poor naming conventions
# - Models with incomplete or unreliable metadata
#
EXCLUDED_MODEL_NAMES = {
    # Generic model names without specific versions
    "gpt",                    # Too generic - use specific versions like "gpt-4"
    "claude",                 # Too generic - use specific versions like "claude-3-opus"
    "llama",                  # Too generic - use specific versions like "llama-3.1-405b"
    "gemini",                 # Too generic - use specific versions like "gemini-1.5-pro"
    "mistral",                # Too generic - use specific versions like "mistral-7b"
    "deepseek",
    
    # Add more exclusions here as needed...
    # "specific_model_name_to_exclude",
}
# =============================================================================


def should_exclude_model(model_name: str) -> bool:
    """
    Check if a model should be excluded from data collection.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model should be excluded, False otherwise
    """
    if not model_name:
        return True  # Exclude models with no name
    
    # Case-insensitive matching against exclusion list
    normalized_name = model_name.strip().lower()
    return normalized_name in EXCLUDED_MODEL_NAMES or normalized_name == ""


def filter_excluded_models(models: list, logger=None) -> tuple:
    """
    Filter out excluded models from a collection.
    
    Args:
        models: List of models to filter
        logger: Optional logger for reporting exclusions
        
    Returns:
        Tuple of (filtered_models, excluded_count, exclusions_detail)
    """
    filtered_models = []
    excluded_models = []
    
    for model in models:
        if should_exclude_model(model.name):
            excluded_models.append(model.name)
            if logger:
                logger.debug(f"Excluded model: '{model.name}' (matches exclusion criteria)")
        else:
            filtered_models.append(model)
    
    # Group exclusions by reason for better reporting
    exclusions_detail = {}
    for excluded_name in excluded_models:
        if not excluded_name or excluded_name.strip() == "":
            reason = "empty/null name"
        else:
            normalized = excluded_name.strip().lower()
            if normalized in EXCLUDED_MODEL_NAMES:
                reason = f"matches '{normalized}'"
            else:
                reason = "other"
        
        if reason not in exclusions_detail:
            exclusions_detail[reason] = []
        exclusions_detail[reason].append(excluded_name)
    
    return filtered_models, len(excluded_models), exclusions_detail


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_scraper(scraper_class, scraper_name: str, storage: JSONStorage) -> bool:
    """Run a single scraper and save results.
    
    Args:
        scraper_class: The scraper class to instantiate
        scraper_name: Name for logging and file storage
        storage: Storage instance for saving data
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Running {scraper_name} scraper...")
        scraper = scraper_class()
        
        # Scrape models
        collection = scraper.scrape_models()
        
        if not collection.models:
            logger.warning(f"{scraper_name} scraper returned no models")
            return False
        
        # Apply name normalization to all models
        normalization_stats = {"original": len(collection.models), "normalized": 0, "conflicts": 0}
        normalized_models = []
        name_mapping = {}  # Track original -> normalized name mapping
        
        for model in collection.models:
            if model.name:
                original_name = model.name
                normalized_name = normalize_model_name(original_name)
                
                # Create a new model instance with normalized name
                normalized_model = model.model_copy()
                normalized_model.name = normalized_name
                normalized_models.append(normalized_model)
                
                # Track normalization for reporting
                if original_name != normalized_name:
                    normalization_stats["normalized"] += 1
                    name_mapping[original_name] = normalized_name
                    logger.info(f"Normalized: {original_name} -> {normalized_name}")
            else:
                # Keep models without names as-is
                normalized_models.append(model)
        
        # Filter out excluded models (after normalization, before saving)
        filtered_models, excluded_count, exclusions_detail = filter_excluded_models(
            normalized_models, logger
        )
        
        # Update collection with filtered models
        collection.models = filtered_models
        
        # Report exclusions
        if excluded_count > 0:
            logger.info(f"{scraper_name}: Excluded {excluded_count} models based on exclusion criteria")
            for reason, excluded_names in exclusions_detail.items():
                logger.info(f"{scraper_name}: Excluded {len(excluded_names)} models ({reason}): {excluded_names[:5]}")
                if len(excluded_names) > 5:
                    logger.info(f"{scraper_name}: ... and {len(excluded_names) - 5} more")
        
        # Save to storage (scraped stage)
        filename = f"{scraper_name.lower().replace(' ', '_')}_models"
        storage.save_models(collection, filename, stage="scraped")
        
        logger.info(f"{scraper_name}: Successfully scraped {len(collection.models)} models after filtering")
        
        # Log processing statistics
        if normalization_stats["normalized"] > 0:
            logger.info(f"{scraper_name}: Normalized {normalization_stats['normalized']} model names")
        
        if excluded_count > 0:
            original_count = normalization_stats["original"]
            logger.info(f"{scraper_name}: Filtered {original_count} -> {len(collection.models)} models ({excluded_count} excluded)")
        
        # Log summary statistics
        above_threshold = len([m for m in collection.models if m.training_flop and m.training_flop >= 1e25])
        developers = len(set(m.developer for m in collection.models))
        
        logger.info(f"{scraper_name}: {above_threshold} models above 1e25 FLOP threshold")
        logger.info(f"{scraper_name}: {developers} unique developers")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to run {scraper_name} scraper: {e}")
        return False


def main():
    """Main entry point for unified data refresh."""
    parser = argparse.ArgumentParser(
        description="Refresh all model data by running all available scrapers"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--scrapers",
        nargs="+",
        choices=["lmarena", "openlm_web", "all"],
        default=["all"],
        help="Which scrapers to run (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for processed data (default: data)"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting unified data refresh...")
    
    # Initialize storage
    storage = JSONStorage(args.output_dir)
    
    # Define available scrapers
    available_scrapers = {
        "lmarena": (LMArenaScraper, "LMArena"),
        "openlm_web": (OpenLMArenaWebScraper, "OpenLM Arena"),
    }
    
    # Determine which scrapers to run
    if "all" in args.scrapers:
        scrapers_to_run = available_scrapers.keys()
    else:
        scrapers_to_run = args.scrapers
    
    # Run scrapers
    results = {}
    total_models = 0
    
    for scraper_key in scrapers_to_run:
        if scraper_key in available_scrapers:
            scraper_class, scraper_name = available_scrapers[scraper_key]
            success = run_scraper(scraper_class, scraper_name, storage)
            results[scraper_name] = success
            
            if success:
                # Count models for summary
                try:
                    filename = f"{scraper_key}_models"
                    collection = storage.load_models(filename, stage="scraped")
                    if collection:
                        total_models += len(collection.models)
                except Exception:
                    pass
        else:
            logger.warning(f"Unknown scraper: {scraper_key}")
    
    # Print summary
    successful_scrapers = [name for name, success in results.items() if success]
    failed_scrapers = [name for name, success in results.items() if not success]
    
    print("\n" + "="*60)
    print("DATA REFRESH SUMMARY")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total models collected: {total_models}")
    print(f"Successful scrapers ({len(successful_scrapers)}): {', '.join(successful_scrapers)}")
    
    if failed_scrapers:
        print(f"Failed scrapers ({len(failed_scrapers)}): {', '.join(failed_scrapers)}")
    
    print("\nRaw scraped data saved to data/scraped/")
    print("Next steps:")
    print("  1. Apply FLOP estimates: python scripts/estimate_flops.py")
    print("  2. Query results: python scripts/query_models.py")
    print("="*60)
    
    # Exit with error code if any scrapers failed
    if failed_scrapers:
        logger.error(f"{len(failed_scrapers)} scrapers failed")
        sys.exit(1)
    
    logger.info("Data refresh completed successfully")


if __name__ == "__main__":
    main()