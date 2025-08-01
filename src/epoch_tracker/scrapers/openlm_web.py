"""OpenLM Chatbot Arena scraper for leaderboard data."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from .base import BaseScraper, ScraperError, ScraperNetworkError
from ..models import Model, ModelCollection


class OpenLMArenaWebScraper(BaseScraper):
    """Web scraper for OpenLM Chatbot Arena leaderboard."""
    
    def __init__(self):
        super().__init__(name="openlm_arena", base_url="https://openlm.ai")
        self.leaderboard_url = "https://openlm.ai/chatbot-arena/"
        # Initialize HTTP session
        self.http = requests.Session()
        self.http.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def scrape_models(self) -> ModelCollection:
        """Scrape models from OpenLM Chatbot Arena."""
        self.logger.info("Starting OpenLM Arena web scraping...")
        
        try:
            response = self.http.get(self.leaderboard_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            models = self._parse_leaderboard_table(soup)
            
            self.logger.info(f"Successfully scraped {len(models)} models from OpenLM Arena")
            
            return ModelCollection(
                models=models,
                source="openlm_arena"
            )
            
        except Exception as e:
            raise ScraperNetworkError(f"Failed to scrape OpenLM Arena: {e}")
    
    def _parse_leaderboard_table(self, soup: BeautifulSoup) -> List[Model]:
        """Parse the leaderboard table from the HTML."""
        models = []
        
        # Find the main table
        table = soup.find('table')
        if not table:
            self.logger.warning("No table found in OpenLM Arena page")
            return models
            
        # Get header row to understand column structure
        header_row = table.find('tr')
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])] if header_row else []
        
        self.logger.debug(f"Table headers: {headers}")
        
        # Expected columns: ['', 'Model', 'Arena Elo', 'Coding', 'Vision', 'AAI', 'MMLU-Pro', 'Votes', 'Organization', 'License']
        header_map = {}
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if 'model' in header_lower:
                header_map['model'] = i
            elif 'arena elo' in header_lower or 'elo' in header_lower:
                header_map['arena_elo'] = i
            elif 'coding' in header_lower:
                header_map['coding'] = i
            elif 'vision' in header_lower:
                header_map['vision'] = i
            elif 'aai' in header_lower:
                header_map['aai'] = i
            elif 'mmlu' in header_lower:
                header_map['mmlu_pro'] = i
            elif 'votes' in header_lower:
                header_map['votes'] = i
            elif 'organization' in header_lower or 'org' in header_lower:
                header_map['organization'] = i
            elif 'license' in header_lower:
                header_map['license'] = i
        
        # Parse data rows
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:  # Need at least rank and model
                continue
                
            try:
                model = self._parse_model_row(cells, header_map)
                if model:
                    models.append(model)
            except Exception as e:
                self.logger.warning(f"Error parsing row: {e}")
                continue
                
        return models
    
    def _parse_model_row(self, cells: List, header_map: dict) -> Optional[Model]:
        """Parse a single model row."""
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        # Get model name
        model_idx = header_map.get('model', 1)  # Default to index 1
        if model_idx >= len(cell_texts) or not cell_texts[model_idx]:
            return None
            
        model_name = cell_texts[model_idx]
        
        # Get other fields
        arena_elo = self._get_cell_value(cell_texts, header_map.get('arena_elo', 2))
        coding_score = self._get_cell_value(cell_texts, header_map.get('coding', 3))
        vision_score = self._get_cell_value(cell_texts, header_map.get('vision', 4))
        aai_score = self._get_cell_value(cell_texts, header_map.get('aai', 5))
        mmlu_score = self._get_cell_value(cell_texts, header_map.get('mmlu_pro', 6))
        votes = self._get_cell_value(cell_texts, header_map.get('votes', 7))
        organization = self._get_cell_value(cell_texts, header_map.get('organization', 8))
        license_type = self._get_cell_value(cell_texts, header_map.get('license', 9))
        
        # Create metadata
        metadata = {
            'arena_elo': self._parse_numeric(arena_elo),
            'coding_score': self._parse_numeric(coding_score),
            'vision_score': self._parse_numeric(vision_score),
            'aai_score': self._parse_numeric(aai_score),
            'mmlu_pro_score': self._parse_numeric(mmlu_score),
            'votes': self._parse_numeric(votes),
            'license': license_type,
            'source': 'openlm_arena'
        }
        
        # Clean up metadata (remove None values)
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Create model
        model = Model(
            name=model_name,
            developer=organization or 'Unknown',
            metadata=metadata
        )
        
        # Add benchmarks
        if arena_elo:
            arena_elo_num = self._parse_numeric(arena_elo)
            if arena_elo_num:
                model.add_benchmark("openlm_arena_elo", arena_elo_num)
                
        if coding_score:
            coding_num = self._parse_numeric(coding_score)
            if coding_num:
                model.add_benchmark("coding_score", coding_num)
                
        if vision_score:
            vision_num = self._parse_numeric(vision_score)
            if vision_num:
                model.add_benchmark("vision_score", vision_num)
        
        # Add source
        model.add_source(self.leaderboard_url, "OpenLM Chatbot Arena leaderboard")
        
        return model
    
    def _get_cell_value(self, cells: List[str], index: Optional[int]) -> Optional[str]:
        """Get cell value by index, return None if index is invalid."""
        if index is None or index >= len(cells):
            return None
        value = cells[index].strip()
        return value if value else None
    
    def _parse_numeric(self, value: Optional[str]) -> Optional[float]:
        """Parse a numeric value from string."""
        if not value:
            return None
        
        # Remove commas and other formatting
        cleaned = value.replace(',', '').replace('+', '').strip()
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    
    def save_to_csv(self, models: List[Model], output_path: Path):
        """Save scraped models to CSV format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Rank', 'Model', 'Arena Elo', 'Coding', 'Vision', 'AAI', 
                'MMLU-Pro', 'Votes', 'Organization', 'License'
            ])
            
            # Write model data
            for i, model in enumerate(models, 1):
                arena_elo = model.benchmarks.get('openlm_arena_elo', '')
                coding = model.benchmarks.get('coding_score', '')
                vision = model.benchmarks.get('vision_score', '')
                aai = model.metadata.get('aai_score', '')
                mmlu = model.metadata.get('mmlu_pro_score', '')
                votes = model.metadata.get('votes', '')
                organization = model.developer
                license_type = model.metadata.get('license', '')
                
                writer.writerow([
                    i, model.name, arena_elo, coding, vision, aai,
                    mmlu, votes, organization, license_type
                ])
                
        self.logger.info(f"Saved {len(models)} models to {output_path}")