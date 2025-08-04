#!/usr/bin/env python3
"""Test script to verify the new configurable scrapers work correctly."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.epoch_tracker.scrapers import (
    LMArenaScraper, 
    OpenLMArenaWebScraper,
    create_scraper
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_lmarena_scrapers():
    """Compare old LMArenaScraper with new ConfigurableScraper."""
    logger.info("\n" + "="*60)
    logger.info("Testing LMArena scrapers")
    logger.info("="*60)
    
    # Test old scraper
    logger.info("Testing old LMArenaScraper...")
    try:
        old_scraper = LMArenaScraper()
        old_collection = old_scraper.scrape_models()
        old_models = old_collection.models
        logger.info(f"Old scraper: Found {len(old_models)} models")
        if old_models:
            logger.info(f"Sample model: {old_models[0].name}")
            logger.info(f"Developer: {old_models[0].developer}")
            if old_models[0].benchmarks:
                logger.info(f"Benchmarks: {old_models[0].benchmarks}")
    except Exception as e:
        logger.error(f"Old scraper failed: {e}")
        old_models = []
    
    # Test new configurable scraper
    logger.info("\nTesting new ConfigurableScraper...")
    try:
        config_path = Path("configs/scrapers/lmarena.json")
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return
            
        new_scraper = create_scraper(config_path)
        new_models = new_scraper.scrape()
        logger.info(f"New scraper: Found {len(new_models)} models")
        if new_models:
            logger.info(f"Sample model: {new_models[0].name}")
            logger.info(f"Developer: {new_models[0].developer}")
            if new_models[0].benchmarks:
                logger.info(f"Benchmarks: {new_models[0].benchmarks}")
    except Exception as e:
        logger.error(f"New scraper failed: {e}")
        import traceback
        traceback.print_exc()
        new_models = []
    
    # Compare results
    if old_models and new_models:
        logger.info(f"\nComparison:")
        logger.info(f"Old scraper models: {len(old_models)}")
        logger.info(f"New scraper models: {len(new_models)}")
        
        old_names = {m.name for m in old_models}
        new_names = {m.name for m in new_models}
        
        if old_names == new_names:
            logger.info("✓ Model names match perfectly!")
        else:
            missing_in_new = old_names - new_names
            extra_in_new = new_names - old_names
            if missing_in_new:
                logger.warning(f"Missing in new: {list(missing_in_new)[:5]}")
            if extra_in_new:
                logger.warning(f"Extra in new: {list(extra_in_new)[:5]}")


def test_openlm_scrapers():
    """Compare old OpenLMArenaWebScraper with new ConfigurableScraper."""
    logger.info("\n" + "="*60)
    logger.info("Testing OpenLM Arena scrapers")
    logger.info("="*60)
    
    # Test old scraper
    logger.info("Testing old OpenLMArenaWebScraper...")
    try:
        old_scraper = OpenLMArenaWebScraper()
        old_collection = old_scraper.scrape_models()
        old_models = old_collection.models
        logger.info(f"Old scraper: Found {len(old_models)} models")
        if old_models:
            logger.info(f"Sample model: {old_models[0].name}")
            logger.info(f"Developer: {old_models[0].developer}")
            if old_models[0].benchmarks:
                logger.info(f"Benchmarks: {old_models[0].benchmarks}")
    except Exception as e:
        logger.error(f"Old scraper failed: {e}")
        old_models = []
    
    # Test new configurable scraper
    logger.info("\nTesting new ConfigurableScraper...")
    try:
        config_path = Path("configs/scrapers/openlm_arena.json")
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return
            
        new_scraper = create_scraper(config_path)
        new_models = new_scraper.scrape()
        logger.info(f"New scraper: Found {len(new_models)} models")
        if new_models:
            logger.info(f"Sample model: {new_models[0].name}")
            logger.info(f"Developer: {new_models[0].developer}")
            if new_models[0].benchmarks:
                logger.info(f"Benchmarks: {new_models[0].benchmarks}")
    except Exception as e:
        logger.error(f"New scraper failed: {e}")
        import traceback
        traceback.print_exc()
        new_models = []
    
    # Compare results
    if old_models and new_models:
        logger.info(f"\nComparison:")
        logger.info(f"Old scraper models: {len(old_models)}")
        logger.info(f"New scraper models: {len(new_models)}")
        
        old_names = {m.name for m in old_models}
        new_names = {m.name for m in new_models}
        
        if old_names == new_names:
            logger.info("✓ Model names match perfectly!")
        else:
            missing_in_new = old_names - new_names
            extra_in_new = new_names - old_names
            if missing_in_new:
                logger.warning(f"Missing in new: {list(missing_in_new)[:5]}")
            if extra_in_new:
                logger.warning(f"Extra in new: {list(extra_in_new)[:5]}")


def main():
    """Run all tests."""
    logger.info("Starting configurable scraper tests...")
    
    # Test LMArena
    test_lmarena_scrapers()
    
    # Test OpenLM Arena (may fail due to network)
    # Uncomment to test web scraping
    # test_openlm_scrapers()
    
    logger.info("\n" + "="*60)
    logger.info("Tests complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()