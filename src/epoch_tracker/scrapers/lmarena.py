"""LMArena scraper that reads from manually collected CSV files."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..models import Model, ModelCollection
from ..scrapers.base import BaseScraper

logger = logging.getLogger(__name__)


class LMArenaScraper(BaseScraper):
    """Scraper for manually collected LMArena leaderboard data."""
    
    def __init__(self):
        super().__init__(name="lmarena")
        # Use default data directory
        self.data_dir = Path("data/raw/lmarena/text")
            
    def scrape_models(self) -> ModelCollection:
        """Read models from CSV files (manual or automated)."""
        models = []
        csv_files = []
        
        # First priority: manual data collection file
        manual_file = self.data_dir / "Manual Data Collection - lmarena-text.csv"
        if manual_file.exists():
            csv_files.append(manual_file)
            
        # Second priority: automated data files (most recent first)
        auto_files = sorted(
            self.data_dir.glob("lmarena_auto_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        csv_files.extend(auto_files)
        
        if not csv_files:
            logger.warning(f"No LMArena data files found in: {self.data_dir}")
            return ModelCollection(
                models=models,
                source="lmarena"
            )
            
        # Read from the first available file (manual takes precedence)
        csv_file = csv_files[0]
        logger.info(f"Reading LMArena data from: {csv_file}")
        
        # Log if we're using automated data
        if 'auto' in csv_file.name:
            logger.info("Using automated scrape data (manual file not found)")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    model = self._parse_model(row)
                    if model:
                        models.append(model)
                        
            logger.info(f"Successfully scraped {len(models)} models from LMArena")
            
            # Add metadata about data source
            file_mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
            logger.info(f"Data file last modified: {file_mtime}")
            
        except Exception as e:
            logger.error(f"Error reading LMArena CSV: {e}")
            
        return ModelCollection(
            models=models,
            source="lmarena"
        )
        
    def _parse_model(self, row: dict) -> Optional[Model]:
        """Parse a single model from CSV row."""
        try:
            # Extract model information
            model_name = row.get('Model', '').strip()
            organization = row.get('Organization', '').strip()
            score = float(row.get('Score', 0))
            rank = row.get('Rank (UB)', '').strip()
            license_type = row.get('License', '').strip()
            
            if not model_name:
                return None
                
            # Add metadata to the model's metadata field
            metadata = {
                'lmarena_score': score,
                'lmarena_rank': rank,
                'license': license_type,
                'source': 'lmarena'
            }
            
            # Create model instance
            model = Model(
                name=model_name,
                developer=organization,  # Changed from organization to developer
                release_date=None,  # Not available in CSV
                parameters=None,    # Not available in CSV
                training_flop=None,  # Will be estimated later
                metadata=metadata
            )
            
            # Add LMArena score as a benchmark
            model.add_benchmark("lmarena_score", score)
            
            # Add source
            model.add_source("LMArena Manual Data Collection", "CSV file with manually collected leaderboard data")
            
            return model
            
        except Exception as e:
            logger.warning(f"Error parsing model from row {row}: {e}")
            return None