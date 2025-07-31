#!/usr/bin/env python3
"""
Fetch OpenLM Arena leaderboard data and save to CSV.

This script scrapes the latest OpenLM Arena leaderboard data and saves it
in CSV format for manual inspection or automated processing.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from epoch_tracker.scrapers.openlm_web import OpenLMArenaWebScraper


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point for fetching OpenLM Arena data."""
    parser = argparse.ArgumentParser(
        description="Fetch OpenLM Arena leaderboard data and save to CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path (default: data/raw/openlm_arena/openlm_<timestamp>.csv)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting OpenLM Arena data fetch...")
    
    # Initialize scraper
    scraper = OpenLMArenaWebScraper()
    
    try:
        # Scrape models
        collection = scraper.scrape_models()
        
        if not collection.models:
            logger.error("No models scraped from OpenLM Arena")
            return 1
            
        logger.info(f"Successfully scraped {len(collection.models)} models")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Default path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("data/raw/openlm_arena")
            output_path = output_dir / f"openlm_{timestamp}.csv"
            
        # Save to CSV
        scraper.save_to_csv(collection.models, output_path)
        logger.info(f"Data saved to: {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("OPENLM ARENA DATA FETCH SUMMARY")
        print(f"{'='*60}")
        print(f"Models scraped: {len(collection.models)}")
        print(f"Output file: {output_path}")
        
        # Show top models by ELO
        models_with_elo = [m for m in collection.models 
                          if m.benchmarks.get('openlm_arena_elo')]
        
        if models_with_elo:
            sorted_models = sorted(
                models_with_elo,
                key=lambda m: m.benchmarks.get('openlm_arena_elo', 0),
                reverse=True
            )
            
            print(f"\nTop 10 models by Arena ELO:")
            for i, model in enumerate(sorted_models[:10], 1):
                elo = model.benchmarks.get('openlm_arena_elo')
                coding = model.benchmarks.get('coding_score', 'N/A')
                print(f"{i:2d}. {model.name:<40} | ELO: {elo:<4} | Org: {model.developer}")
            
            # Show statistics
            elos = [m.benchmarks.get('openlm_arena_elo') for m in models_with_elo]
            print(f"\nELO Statistics:")
            print(f"  Models with ELO: {len(elos)}")
            print(f"  Highest ELO: {max(elos)}")
            print(f"  Lowest ELO: {min(elos)}")
            print(f"  Average ELO: {sum(elos)/len(elos):.1f}")
        
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to fetch OpenLM Arena data: {e}")
        return 1
    finally:
        scraper.http.close()


if __name__ == "__main__":
    sys.exit(main())