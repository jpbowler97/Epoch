#!/usr/bin/env python3
"""
Unified data refresh script for all scrapers.

This script runs all available scrapers and prepares data for querying.
It serves as the main entry point for data collection in the Epoch tracker.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.scrapers import create_scraper
from epoch_tracker.scrapers.claude_service import ClaudeScraperWithFallback
from epoch_tracker.storage import JSONStorage
from epoch_tracker.utils.model_names import normalize_model_name
from epoch_tracker.utils.developer_blacklist import DeveloperBlacklist


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


def run_configurable_scraper(config_path: Path, scraper_name: str, storage: JSONStorage) -> bool:
    """Run a configurable scraper from JSON config.
    
    Args:
        config_path: Path to scraper JSON configuration
        scraper_name: Name for logging and file storage
        storage: Storage instance for saving data
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Running {scraper_name} scraper...")
        
        # Create scraper from JSON config
        scraper = create_scraper(config_path)
        
        # Scrape models
        models = scraper.scrape()
        
        if not models:
            logger.warning(f"{scraper_name} scraper returned no models")
            return False
        
        # Apply name normalization to all models
        normalization_stats = {"original": len(models), "normalized": 0, "conflicts": 0}
        normalized_models = []
        name_mapping = {}  # Track original -> normalized name mapping
        
        for model in models:
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
        
        # Create ModelCollection for storage
        from epoch_tracker.models import ModelCollection
        collection = ModelCollection(models=filtered_models, source=scraper_name)
        
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


def discover_scrapers(config_dir: Path) -> dict:
    """Discover all available scraper configurations.
    
    Args:
        config_dir: Directory containing scraper JSON configurations
        
    Returns:
        Dictionary mapping scraper names to config paths
    """
    scrapers = {}
    
    if config_dir.exists():
        for config_file in config_dir.glob("*.json"):
            scraper_name = config_file.stem
            scrapers[scraper_name] = config_file
    
    return scrapers


def check_for_new_developers(storage: JSONStorage, successful_scrapers: list, logger) -> list:
    """Check for new developers that aren't in the blacklist configuration.
    
    Args:
        storage: JSONStorage instance for loading models
        successful_scrapers: List of scrapers that ran successfully
        logger: Logger instance for reporting
        
    Returns:
        List of new developers found that need review
    """
    try:
        # Load developer blacklist
        blacklist = DeveloperBlacklist()
        known_developers = set(blacklist.config.get('blacklist', {}).keys())
        pending_developers = set(blacklist.get_pending_review())
        
        # Collect all developers from scraped models
        all_developers = set()
        
        for scraper_name in successful_scrapers:
            try:
                filename = f"{scraper_name}_models"
                collection = storage.load_models(filename, stage="scraped")
                if collection:
                    for model in collection.models:
                        if model.developer and model.developer.strip():
                            all_developers.add(model.developer.strip())
            except Exception as e:
                logger.warning(f"Could not check developers from {scraper_name}: {e}")
        
        # Find new developers not in blacklist or pending
        new_developers = all_developers - known_developers - pending_developers
        
        if new_developers:
            logger.warning(f"Found {len(new_developers)} new developers that need blacklist review:")
            for dev in sorted(new_developers):
                logger.warning(f"  - {dev}")
                blacklist.add_pending_developer(dev)
            
            # Save updated blacklist with pending developers
            blacklist.save_config()
            logger.info(f"Added {len(new_developers)} developers to pending review list")
            
            return list(new_developers)
        else:
            logger.info("No new developers found - all are already in blacklist configuration")
            return []
            
    except Exception as e:
        logger.error(f"Failed to check for new developers: {e}")
        return []


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
        help="Which scrapers to run (default: all discovered scrapers)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/scrapers"),
        help="Directory containing scraper JSON configurations (default: configs/scrapers)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for processed data (default: data)"
    )
    parser.add_argument(
        "--skip-missing-files",
        action="store_true",
        help="Skip scrapers whose source files don't exist (useful for testing)"
    )
    parser.add_argument(
        "--update-claude-sites",
        action="store_true",
        help="Update JavaScript-heavy sites via Claude Code before running scrapers"
    )
    parser.add_argument(
        "--force-claude-update",
        action="store_true",
        help="Force Claude to update all configured sites regardless of update frequency"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting unified data refresh...")
    
    # Initialize Claude scraper and check data freshness
    claude_scraper = ClaudeScraperWithFallback()
    
    # Always show Claude data status at start
    print("🔍 Checking JavaScript-heavy benchmark sites...")
    freshness = claude_scraper.check_files_freshness(max_age_hours=24)
    claude_sites = list(freshness.keys())
    
    if claude_sites:
        fresh_sites = [site for site, fresh in freshness.items() if fresh]
        stale_sites = [site for site, fresh in freshness.items() if not fresh]
        
        print(f"📊 Claude-managed sites ({len(claude_sites)}): {', '.join(claude_sites)}")
        
        if fresh_sites:
            print(f"✅ Fresh data (<24h): {', '.join(fresh_sites)}")
        if stale_sites:
            print(f"⏰ Stale data (>24h): {', '.join(stale_sites)}")
            
        # Show last update times
        for site in claude_sites:
            try:
                file_path = Path(f"data/benchmark_files/{site.title().replace('_', '')}.html")
                if file_path.exists():
                    from datetime import datetime
                    import os
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age_str = mtime.strftime("%Y-%m-%d %H:%M")
                    print(f"  📅 {site}: last updated {age_str}")
            except Exception:
                print(f"  📅 {site}: no data file found")
    else:
        print("ℹ️  No Claude-managed sites configured")
    
    print()  # Add spacing
    
    if args.update_claude_sites or args.force_claude_update:
        logger.info("Updating JavaScript-heavy sites via Claude Code...")
        
        if args.force_claude_update:
            # Force update all configured sites
            sites_to_update = list(claude_scraper.config["claude_scrapers"].keys())
        else:
            # Only update sites that need it
            sites_to_update = claude_scraper.get_sites_needing_update()
        
        if sites_to_update:
            command = claude_scraper.show_manual_instructions(sites_to_update)
            logger.warning("Manual Claude Code execution required - see instructions above")
            print("\nWaiting for you to run the Claude command...")
            input("Press Enter after Claude has completed the updates to continue...")
            
            # Mark as updated
            claude_scraper.mark_updated(sites_to_update)
            logger.info("Marked sites as updated, continuing with data collection...")
        else:
            logger.info("No Claude-managed sites need updating")
    else:
        # Show guidance for updating stale Claude data if needed
        stale_sites = [site for site, fresh in freshness.items() if not fresh]
        
        if stale_sites:
            print("💡 To update JavaScript-heavy sites with fresh data:")
            command = claude_scraper.show_manual_instructions(stale_sites)
            print("   Then re-run: python scripts/run.py collect-all")
            print("   (Continuing with existing files for now...)")
            print()
    
    # Initialize storage
    storage = JSONStorage(args.output_dir)
    
    # Discover available scrapers
    available_scrapers = discover_scrapers(args.config_dir)
    
    if not available_scrapers:
        logger.error(f"No scraper configurations found in {args.config_dir}")
        sys.exit(1)
    
    logger.info(f"Discovered {len(available_scrapers)} scrapers: {', '.join(available_scrapers.keys())}")
    
    # Determine which scrapers to run
    if args.scrapers:
        scrapers_to_run = {}
        for scraper_name in args.scrapers:
            if scraper_name in available_scrapers:
                scrapers_to_run[scraper_name] = available_scrapers[scraper_name]
            else:
                logger.warning(f"Unknown scraper: {scraper_name}")
    else:
        scrapers_to_run = available_scrapers
    
    # Check for missing source files if skip flag is set
    if args.skip_missing_files:
        valid_scrapers = {}
        for scraper_name, config_path in scrapers_to_run.items():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                source_path = config.get('source', {}).get('path')
                if source_path:
                    source_file = Path(source_path)
                    if not source_file.exists():
                        logger.info(f"Skipping {scraper_name}: source file {source_file} does not exist")
                        continue
                valid_scrapers[scraper_name] = config_path
            except Exception as e:
                logger.warning(f"Could not check source file for {scraper_name}: {e}")
                valid_scrapers[scraper_name] = config_path
        scrapers_to_run = valid_scrapers
    
    # Run scrapers
    results = {}
    total_models = 0
    
    for scraper_name, config_path in scrapers_to_run.items():
        success = run_configurable_scraper(config_path, scraper_name, storage)
        results[scraper_name] = success
        
        if success:
            # Count models for summary
            try:
                filename = f"{scraper_name}_models"
                collection = storage.load_models(filename, stage="scraped")
                if collection:
                    total_models += len(collection.models)
            except Exception:
                pass
    
    # Print summary
    successful_scrapers = [name for name, success in results.items() if success]
    failed_scrapers = [name for name, success in results.items() if not success]
    
    # Check for new developers that need blacklist review
    new_developers = check_for_new_developers(storage, successful_scrapers, logger)
    
    print("\n" + "="*60)
    print("DATA REFRESH SUMMARY")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total models collected: {total_models}")
    
    # Show per-scraper breakdown if verbose
    if args.verbose and successful_scrapers:
        print(f"\nScraper breakdown:")
        for scraper_name in successful_scrapers:
            try:
                filename = f"{scraper_name}_models"
                collection = storage.load_models(filename, stage="scraped")
                if collection:
                    model_count = len(collection.models)
                    print(f"  • {scraper_name}: {model_count} models")
            except Exception:
                print(f"  • {scraper_name}: ? models (could not load)")
    
    print(f"\nSuccessful scrapers ({len(successful_scrapers)}): {', '.join(successful_scrapers)}")
    
    if failed_scrapers:
        print(f"Failed scrapers ({len(failed_scrapers)}): {', '.join(failed_scrapers)}")
    
    # Show new developers that need review
    if new_developers:
        print(f"\n⚠️  NEW DEVELOPERS FOUND ({len(new_developers)}) - REVIEW REQUIRED:")
        for dev in sorted(new_developers):
            print(f"  • {dev}")
        print("\nTo review these developers, run:")
        print("  python scripts/run.py review-developers")
    
    print("\nRaw scraped data saved to data/scraped/")
    
    # Show Claude data status in summary
    if claude_sites:
        fresh_count = len([site for site, fresh in freshness.items() if fresh])
        stale_count = len(claude_sites) - fresh_count
        print(f"Claude-managed sites: {fresh_count} fresh, {stale_count} stale")
        if stale_count > 0:
            print("  💡 For freshest data: python scripts/run.py update-claude")
    
    print("\nNext steps:")
    print("  1. Apply FLOP estimates: python scripts/run.py estimate-flops --update")
    print("  2. Query results: python scripts/run.py query --above-threshold")
    print("="*60)
    
    # Exit with error code if any scrapers failed
    if failed_scrapers:
        logger.error(f"{len(failed_scrapers)} scrapers failed")
        sys.exit(1)
    
    logger.info("Data refresh completed successfully")


if __name__ == "__main__":
    main()