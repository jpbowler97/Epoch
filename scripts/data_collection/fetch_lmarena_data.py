#!/usr/bin/env python3
"""
Fetch LMArena leaderboard data and save to CSV.

This script automates the process of fetching the latest LMArena leaderboard
data and saving it in the format expected by the manual data collection pipeline.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.epoch_tracker.scrapers.lmarena_web import LMArenaWebScraper


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point for fetching LMArena data."""
    parser = argparse.ArgumentParser(
        description="Fetch LMArena leaderboard data and save to CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path (default: data/raw/lmarena/text/lmarena_auto_<timestamp>.csv)"
    )
    parser.add_argument(
        "--update-manual",
        action="store_true",
        help="Update the manual collection file (Manual Data Collection - lmarena-text.csv)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting LMArena data fetch...")
    
    # Initialize scraper
    scraper = LMArenaWebScraper()
    
    try:
        # Scrape models
        collection = scraper.scrape_models()
        
        if not collection.models:
            logger.error("No models scraped from LMArena")
            return 1
            
        logger.info(f"Successfully scraped {len(collection.models)} models")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Default path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("data/raw/lmarena/text")
            output_path = output_dir / f"lmarena_auto_{timestamp}.csv"
            
        # Save to CSV
        scraper.save_to_csv(collection.models, output_path)
        logger.info(f"Data saved to: {output_path}")
        
        # Optionally update the manual collection file
        if args.update_manual:
            manual_path = Path("data/raw/lmarena/text/Manual Data Collection - lmarena-text.csv")
            
            # Backup existing file
            if manual_path.exists():
                backup_path = manual_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                manual_path.rename(backup_path)
                logger.info(f"Backed up existing manual file to: {backup_path}")
                
            # Save as new manual collection
            scraper.save_to_csv(collection.models, manual_path)
            logger.info(f"Updated manual collection file: {manual_path}")
            
        # Print summary
        print(f"\n{'='*60}")
        print("LMARENA DATA FETCH SUMMARY")
        print(f"{'='*60}")
        print(f"Models scraped: {len(collection.models)}")
        print(f"Output file: {output_path}")
        
        # Show top models
        print(f"\nTop 10 models by score:")
        sorted_models = sorted(
            collection.models,
            key=lambda m: m.benchmarks.get('lmarena_score', m.benchmarks.get('lmarena_elo', 0)),
            reverse=True
        )
        
        for i, model in enumerate(sorted_models[:10], 1):
            score = model.benchmarks.get('lmarena_score', model.benchmarks.get('lmarena_elo', 'N/A'))
            print(f"{i:2d}. {model.name:<40} (Score: {score})")
            
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to fetch LMArena data: {e}")
        return 1
    finally:
        scraper.http.close()


if __name__ == "__main__":
    sys.exit(main())