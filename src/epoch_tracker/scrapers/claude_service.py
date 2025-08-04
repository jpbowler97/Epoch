"""
Claude Web Scraper Service

This module provides integration with Claude Code's WebFetch capabilities
to scrape JavaScript-heavy websites that regular Python scrapers cannot handle.
"""

import json
import logging
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class ClaudeWebScraper:
    """Service for using Claude Code to scrape JavaScript-heavy sites"""
    
    def __init__(self, config_path: str = "configs/claude_scraping.yaml"):
        """Initialize the Claude scraper service"""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.benchmark_files_dir = Path("data/benchmark_files")
        self.benchmark_files_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self) -> Dict:
        """Load configuration for Claude-managed scrapers"""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for Claude scrapers"""
        return {
            "claude_scrapers": {
                "superclue": {
                    "url": "https://www.superclueai.com/",
                    "output_file": "data/benchmark_files/SuperCLUE.html",
                    "extraction_prompt": (
                        "Extract the complete leaderboard table with:\n"
                        "- Model names (模型名称)\n"
                        "- Organization (机构)\n"
                        "- Overall score (总分)\n"
                        "- Math, reasoning, code, agents scores\n"
                        "Format as a clean HTML table with proper <table>, <thead>, <tbody> structure"
                    ),
                    "update_frequency": "daily",
                    "last_updated": None
                },
                "physics_iq": {
                    "url": "https://physics-iq.github.io/",
                    "output_file": "data/benchmark_files/PhysicsIQ.html",
                    "extraction_prompt": (
                        "Extract the physics understanding benchmark table with:\n"
                        "- Model names\n"
                        "- Physics IQ scores\n"
                        "- Model types (i2v, multiframe, etc.)\n"
                        "Format as HTML table"
                    ),
                    "update_frequency": "weekly",
                    "last_updated": None
                },
                "olympic_arena": {
                    "url": "https://gair-nlp.github.io/OlympicArena",
                    "output_file": "data/benchmark_files/OlympicArena.html",
                    "extraction_prompt": (
                        "Extract the complete leaderboard with:\n"
                        "- Model names and developers\n"
                        "- Overall scores\n"
                        "- Subject-specific scores (Math, Physics, Chemistry, etc.)\n"
                        "Format as HTML table"
                    ),
                    "update_frequency": "weekly",
                    "last_updated": None
                },
                "video_arena": {
                    "url": "https://artificialanalysis.ai/text-to-video/arena",
                    "output_file": "data/benchmark_files/VideoArena.html",
                    "extraction_prompt": (
                        "Extract the text-to-video model arena leaderboard with:\n"
                        "- Model names\n"
                        "- ELO scores\n"
                        "- Quality ratings\n"
                        "Format as HTML table"
                    ),
                    "update_frequency": "weekly",
                    "last_updated": None
                }
            },
            "settings": {
                "max_retries": 3,
                "timeout_seconds": 120,
                "use_subprocess": True,  # Use CLI vs API
                "auto_confirm": True,
                "verbose": True
            }
        }
    
    def save_config(self):
        """Save current configuration back to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def needs_update(self, scraper_name: str) -> bool:
        """Check if a scraper needs updating based on frequency"""
        scraper_config = self.config["claude_scrapers"].get(scraper_name)
        if not scraper_config:
            logger.warning(f"Scraper {scraper_name} not found in config")
            return False
            
        last_updated = scraper_config.get("last_updated")
        if not last_updated:
            return True  # Never updated
            
        # Parse last updated time
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        
        # Check frequency
        frequency = scraper_config.get("update_frequency", "weekly")
        if frequency == "daily":
            threshold = timedelta(days=1)
        elif frequency == "weekly":
            threshold = timedelta(weeks=1)
        elif frequency == "monthly":
            threshold = timedelta(days=30)
        else:
            threshold = timedelta(weeks=1)  # Default to weekly
            
        return datetime.now() - last_updated > threshold
    
    def get_sites_needing_update(self) -> List[str]:
        """Get list of sites that need updating"""
        sites = []
        for scraper_name in self.config["claude_scrapers"]:
            if self.needs_update(scraper_name):
                sites.append(scraper_name)
        return sites
    
    def generate_instructions(self, scrapers_to_update: List[str]) -> str:
        """Generate detailed instructions for Claude Code"""
        if not scrapers_to_update:
            return ""
            
        instructions = [
            "# Task: Update Complex Benchmark Data",
            "",
            "You need to fetch and save benchmark data from JavaScript-rendered websites.",
            "Use your WebFetch tool to get the rendered content, then save it as HTML files.",
            "",
            "## Instructions:",
            ""
        ]
        
        for i, scraper_name in enumerate(scrapers_to_update, 1):
            scraper = self.config["claude_scrapers"][scraper_name]
            instructions.extend([
                f"### {i}. Update {scraper_name}",
                "",
                f"1. Use WebFetch to get content from: {scraper['url']}",
                "2. Extract the following data:",
                f"   {scraper['extraction_prompt'].replace(chr(10), chr(10) + '   ')}",
                "3. Create a clean HTML file with:",
                "   - DOCTYPE and proper HTML structure",
                "   - A <table> element with the extracted data",
                "   - Proper <thead> and <tbody> sections",
                "   - All column headers and data preserved",
                f"4. Save to: {scraper['output_file']}",
                f"5. Add timestamp comment: <!-- Updated: {datetime.now().isoformat()} -->",
                ""
            ])
        
        instructions.extend([
            "## Output Requirements:",
            "- Each HTML file must be valid HTML with proper structure",
            "- Tables must have clear headers matching the data",
            "- Preserve all numeric scores exactly as found",
            "- Include model names without modification",
            "",
            "## Important:",
            "- If a site requires interaction or returns no data, log the issue",
            "- Create the HTML files even if you only get partial data",
            "- Ensure files are saved in the exact paths specified"
        ])
        
        return "\n".join(instructions)
    
    def save_instructions_for_manual_execution(self, instructions: str) -> str:
        """Save instructions to file and return simple CLI command for user"""
        instructions_file = Path("scripts/claude_instructions/update_benchmarks.md")
        instructions_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the detailed instructions
        instructions_file.write_text(instructions)
        
        # Return simple command for user
        return f"claude code {instructions_file}"
    
    def validate_output(self, scraper_name: str) -> bool:
        """Validate that Claude successfully created the HTML file"""
        scraper = self.config["claude_scrapers"].get(scraper_name)
        if not scraper:
            return False
            
        output_file = Path(scraper["output_file"])
        
        # Check file exists
        if not output_file.exists():
            logger.error(f"Output file {output_file} not found for {scraper_name}")
            return False
        
        # Check file is recent (modified in last 5 minutes)
        mtime = datetime.fromtimestamp(output_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(minutes=5):
            logger.warning(f"Output file {output_file} is not recent (modified {mtime})")
            return False
        
        # Check file has content
        content = output_file.read_text()
        if len(content) < 100:
            logger.error(f"Output file {output_file} is too small ({len(content)} bytes)")
            return False
        
        # Check for table element
        if "<table" not in content.lower():
            logger.error(f"Output file {output_file} does not contain a table element")
            return False
        
        logger.info(f"Validation successful for {scraper_name}")
        return True
    
    def show_manual_instructions(self, scrapers: Optional[List[str]] = None) -> str:
        """Generate instructions and show user how to execute manually"""
        if scrapers is None:
            scrapers = self.get_sites_needing_update()
        
        if not scrapers:
            logger.info("No scrapers need updating")
            return ""
        
        # Generate instructions
        instructions = self.generate_instructions(scrapers)
        
        # Save to file and get simple command
        command = self.save_instructions_for_manual_execution(instructions)
        
        # Show user what to do
        print("\n" + "="*60)
        print("CLAUDE CODE MANUAL UPDATE REQUIRED")
        print("="*60)
        print(f"\nTo update {len(scrapers)} benchmark sites, run this command:\n")
        print(f"  {command}")
        print(f"\nThis will update: {', '.join(scrapers)}")
        print("\nAfter Claude completes the updates, your scrapers will use the fresh data.")
        print("="*60)
        
        return command
    
    def check_files_freshness(self, max_age_hours: int = 24) -> Dict[str, bool]:
        """Check if Claude-managed files are fresh enough"""
        results = {}
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for scraper_name, config in self.config["claude_scrapers"].items():
            if not config.get("enabled", True):
                continue
                
            file_path = Path(config["output_file"])
            if not file_path.exists():
                results[scraper_name] = False
                continue
                
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            results[scraper_name] = mtime > cutoff_time
            
        return results
    
    def mark_updated(self, scraper_names: List[str]):
        """Mark scrapers as updated (call this after Claude execution)"""
        for scraper_name in scraper_names:
            if scraper_name in self.config["claude_scrapers"]:
                self.config["claude_scrapers"][scraper_name]["last_updated"] = datetime.now().isoformat()
        self.save_config()


class ClaudeScraperWithFallback(ClaudeWebScraper):
    """Extended Claude scraper with fallback mechanisms"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = Path("data/benchmark_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cached_file(self, scraper_name: str) -> Optional[Path]:
        """Get path to cached file for a scraper"""
        scraper = self.config["claude_scrapers"].get(scraper_name)
        if not scraper:
            return None
            
        # Check if current file exists and is reasonably recent
        output_file = Path(scraper["output_file"])
        if output_file.exists():
            # Use existing file as cache
            return output_file
            
        # Check dedicated cache
        cache_file = self.cache_dir / f"{scraper_name}.html"
        if cache_file.exists():
            return cache_file
            
        return None
    
    def update_with_fallback(self, scraper_name: str) -> bool:
        """Update scraper with fallback to cached data"""
        try:
            # Try normal update
            if self.update_single(scraper_name):
                return True
                
        except Exception as e:
            logger.error(f"Error updating {scraper_name}: {e}")
        
        # Fall back to cached file
        cached_file = self.get_cached_file(scraper_name)
        if cached_file:
            logger.warning(f"Using cached file for {scraper_name}: {cached_file}")
            
            # Copy cache to expected location if different
            scraper = self.config["claude_scrapers"][scraper_name]
            output_file = Path(scraper["output_file"])
            if cached_file != output_file:
                import shutil
                shutil.copy2(cached_file, output_file)
                
            return True
        
        logger.error(f"No fallback available for {scraper_name}")
        return False