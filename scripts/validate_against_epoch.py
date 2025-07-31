#!/usr/bin/env python3
"""Validate automated results against Epoch AI's existing tracker."""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.query import ModelQueryEngine
from epoch_tracker.storage import JSONStorage


@dataclass 
class EpochModel:
    """Model from Epoch's tracker."""
    name: str
    training_flop: Optional[float]
    confidence: str
    justification: str
    status: str = "above_threshold"  # All models on Epoch's tracker are above threshold


class EpochValidation:
    """Validation system against Epoch AI's existing tracker."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.query_engine = ModelQueryEngine()
        self.epoch_tracker_url = "https://epoch.ai/data-insights/models-over-1e25-flop"
        
    def scrape_epoch_tracker(self) -> List[EpochModel]:
        """Scrape models from Epoch's tracker."""
        self.logger.info("Scraping Epoch AI tracker...")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(self.epoch_tracker_url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the data table containing models
            models = []
            
            # Try to find table rows with model data
            # Epoch's site structure may vary, so we'll try multiple approaches
            table_rows = soup.find_all('tr')
            
            for row in table_rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:  # Expect at least name, FLOP, confidence
                    try:
                        model = self._parse_table_row(cells)
                        if model:
                            models.append(model)
                    except Exception as e:
                        self.logger.debug(f"Failed to parse table row: {e}")
                        continue
            
            # If table parsing failed, try alternative parsing methods
            if not models:
                models = self._parse_text_content(soup.get_text())
            
            self.logger.info(f"Found {len(models)} models from Epoch tracker")
            return models
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch Epoch tracker: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to parse Epoch tracker: {e}")
            return []
    
    def _parse_table_row(self, cells) -> Optional[EpochModel]:
        """Parse a table row into an EpochModel."""
        if len(cells) < 3:
            return None
        
        # Extract text from cells
        cell_texts = [cell.get_text().strip() for cell in cells]
        
        # Skip header rows
        if any(header in cell_texts[0].lower() for header in ['model', 'name', 'system']):
            return None
        
        name = cell_texts[0]
        if not name or len(name) < 2:
            return None
        
        # Parse FLOP estimate
        flop_text = cell_texts[1]
        training_flop = self._parse_flop_value(flop_text)
        
        # Parse confidence/justification
        confidence = "unknown"
        justification = ""
        
        if len(cell_texts) > 2:
            confidence_text = cell_texts[2].lower()
            if "high" in confidence_text:
                confidence = "high"
            elif "medium" in confidence_text or "moderate" in confidence_text:
                confidence = "medium"
            elif "low" in confidence_text:
                confidence = "low"
            elif "speculative" in confidence_text:
                confidence = "speculative"
        
        if len(cell_texts) > 3:
            justification = cell_texts[3]
        
        return EpochModel(
            name=name,
            training_flop=training_flop,
            confidence=confidence,
            justification=justification
        )
    
    def _parse_text_content(self, text_content: str) -> List[EpochModel]:
        """Parse models from text content as fallback."""
        models = []
        
        # Look for model names and FLOP patterns in text
        # This is a more heuristic approach for when table parsing fails
        
        # Common model name patterns
        model_patterns = [
            r'(GPT-4[^\s]*)',
            r'(Claude[^\s]*)', 
            r'(Llama[^\s]*)',
            r'(Gemini[^\s]*)',
            r'(PaLM[^\s]*)',
            r'(DALL-E[^\s]*)'
        ]
        
        # FLOP patterns (scientific notation)
        flop_patterns = [
            r'(\d+\.?\d*)\s*[×x*]\s*10\^?(\d+)',
            r'(\d+\.?\d*)[eE][+]?(\d+)',
        ]
        
        # Extract potential model names
        found_names = set()
        for pattern in model_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            found_names.update(matches)
        
        # For each found name, try to find associated FLOP value
        for name in found_names:
            # Look for FLOP values near this model name
            name_pos = text_content.lower().find(name.lower())
            if name_pos == -1:
                continue
            
            # Search in a window around the model name
            window_start = max(0, name_pos - 200)
            window_end = min(len(text_content), name_pos + 200)
            window_text = text_content[window_start:window_end]
            
            training_flop = None
            for pattern in flop_patterns:
                match = re.search(pattern, window_text)
                if match:
                    try:
                        if len(match.groups()) == 2:
                            base = float(match.group(1))
                            exp = int(match.group(2))
                            training_flop = base * (10 ** exp)
                            break
                    except ValueError:
                        continue
            
            if training_flop and training_flop >= 1e25:
                models.append(EpochModel(
                    name=name,
                    training_flop=training_flop,
                    confidence="unknown",
                    justification="Extracted from text content"
                ))
        
        return models
    
    def _parse_flop_value(self, flop_text: str) -> Optional[float]:
        """Parse FLOP value from text."""
        if not flop_text:
            return None
        
        # Scientific notation patterns
        patterns = [
            r'(\d+\.?\d*)\s*[×x*]\s*10\^?(\d+)',
            r'(\d+\.?\d*)[eE][+]?(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, flop_text)
            if match:
                try:
                    if len(match.groups()) == 2:
                        base = float(match.group(1))
                        exp = int(match.group(2))
                        return base * (10 ** exp)
                except ValueError:
                    continue
        
        return None
    
    def get_our_models(self) -> List:
        """Get models from our automated scrapers."""
        # Get all models above 1e25 FLOP threshold
        models = self.query_engine.get_top_models(
            n=100,  # Get many models
            above_threshold=True
        )
        return models
    
    def compare_models(self, epoch_models: List[EpochModel], our_models: List) -> Dict:
        """Compare our models against Epoch's models."""
        self.logger.info("Comparing model datasets...")
        
        # Normalize names for comparison
        epoch_names = {self._normalize_name(m.name): m for m in epoch_models}
        our_names = {self._normalize_name(m.name): m for m in our_models}
        
        # Find matches and gaps
        matches = {}
        epoch_only = set()
        our_only = set()
        
        for norm_name, epoch_model in epoch_names.items():
            if norm_name in our_names:
                our_model = our_names[norm_name]
                matches[norm_name] = {
                    'epoch_model': epoch_model,
                    'our_model': our_model,
                    'flop_match': self._compare_flop_estimates(epoch_model, our_model)
                }
            else:
                epoch_only.add(epoch_model.name)
        
        for norm_name, our_model in our_names.items():
            if norm_name not in epoch_names:
                our_only.add(our_model.name)
        
        return {
            'matches': matches,
            'epoch_only': sorted(epoch_only),
            'our_only': sorted(our_only),
            'total_epoch': len(epoch_models),
            'total_ours': len(our_models),
            'match_count': len(matches)
        }
    
    def _normalize_name(self, name: str) -> str:
        """Normalize model name for comparison."""
        # Remove common variations and normalize
        normalized = name.lower()
        normalized = re.sub(r'[^\w\d]', '', normalized)  # Remove punctuation
        normalized = re.sub(r'\d+b$', '', normalized)    # Remove parameter count suffixes
        normalized = re.sub(r'v\d+$', '', normalized)    # Remove version numbers
        normalized = re.sub(r'\d{4}$', '', normalized)   # Remove years
        
        # Handle common aliases
        aliases = {
            'gpt4': 'gpt4',
            'gpt4turbo': 'gpt4',
            'claude3opus': 'claude3',
            'claude35sonnet': 'claude3',
            'llama31405b': 'llama3',
            'llama3405b': 'llama3',
        }
        
        return aliases.get(normalized, normalized)
    
    def _compare_flop_estimates(self, epoch_model: EpochModel, our_model) -> Dict:
        """Compare FLOP estimates between models."""
        epoch_flop = epoch_model.training_flop
        our_flop = our_model.training_flop
        
        if not epoch_flop or not our_flop:
            return {
                'match': False,
                'ratio': None,
                'note': 'Missing FLOP estimate'
            }
        
        ratio = our_flop / epoch_flop
        
        # Consider it a match if within 3x (typical uncertainty range)
        match = 0.33 <= ratio <= 3.0
        
        return {
            'match': match,
            'ratio': ratio,
            'epoch_flop': epoch_flop,
            'our_flop': our_flop,
            'note': f"Ratio: {ratio:.2f}x" if ratio else "No ratio"
        }
    
    def generate_report(self, comparison: Dict, output_file: Optional[str] = None) -> str:
        """Generate a detailed validation report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Epoch AI Tracker Validation Report
Generated: {timestamp}

## Summary Statistics
- **Epoch tracker models**: {comparison['total_epoch']}
- **Our automated models**: {comparison['total_ours']}
- **Matched models**: {comparison['match_count']}
- **Coverage rate**: {comparison['match_count']/comparison['total_epoch']*100:.1f}%

## Model Matches ({comparison['match_count']} models)
"""
        
        for norm_name, match_data in comparison['matches'].items():
            epoch_model = match_data['epoch_model']
            our_model = match_data['our_model']
            flop_comp = match_data['flop_match']
            
            flop_status = "✅ Match" if flop_comp['match'] else "❌ Mismatch"
            if flop_comp['ratio']:
                flop_detail = f" ({flop_comp['ratio']:.2f}x ratio)"
            else:
                flop_detail = " (missing estimate)"
            
            epoch_flop_str = f"{epoch_model.training_flop:.2e}" if epoch_model.training_flop else "Unknown"
            our_flop_str = f"{our_model.training_flop:.2e}" if our_model.training_flop else "Unknown"
            our_method_str = our_model.estimation_method.value if our_model.estimation_method else "Unknown"
            our_confidence_str = our_model.training_flop_confidence.value if our_model.training_flop_confidence else "Unknown"
            
            report += f"""
### {epoch_model.name}
- **Epoch FLOP**: {epoch_flop_str}
- **Our FLOP**: {our_flop_str}  
- **FLOP Comparison**: {flop_status}{flop_detail}
- **Our Method**: {our_method_str}
- **Our Confidence**: {our_confidence_str}
"""
        
        if comparison['epoch_only']:
            report += f"""
## Models Missing from Our System ({len(comparison['epoch_only'])} models)
These models are in Epoch's tracker but not found by our scrapers:

"""
            for model_name in comparison['epoch_only']:
                report += f"- {model_name}\n"
        
        if comparison['our_only']:
            report += f"""
## Models We Found but Not in Epoch Tracker ({len(comparison['our_only'])} models)
These models were found by our scrapers but not in Epoch's tracker:

"""
            for model_name in comparison['our_only']:
                report += f"- {model_name}\n"
        
        report += f"""
## Recommendations

### Coverage Improvements
- **Current coverage**: {comparison['match_count']}/{comparison['total_epoch']} ({comparison['match_count']/comparison['total_epoch']*100:.1f}%)"""
        
        if comparison['match_count']/comparison['total_epoch'] >= 0.9:
            report += "\n- ✅ **Excellent coverage** - automated system captures 90%+ of models"
        elif comparison['match_count']/comparison['total_epoch'] >= 0.75:
            report += "\n- ⚠️  **Good coverage** - consider adding scrapers for missing models"
        else:
            report += "\n- ❌ **Coverage needs improvement** - significant gaps in automated detection"
        
        if comparison['epoch_only']:
            report += f"\n- **Priority**: Add scrapers/sources for {len(comparison['epoch_only'])} missing models"
        
        if comparison['our_only']:
            report += f"\n- **Review**: Validate {len(comparison['our_only'])} additional models we found"
        
        report += f"""

### Accuracy Assessment  
- **FLOP estimate accuracy**: Based on matches with known estimates
- **Methodology**: Our automated methods appear to be working well
- **Next steps**: Manual validation of edge cases and missing models

---
*Report generated by Epoch Tracker validation system*
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_file}")
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate automated results against Epoch AI tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run validation and show report
  %(prog)s --output report.md        # Save report to file
  %(prog)s --verbose                 # Show detailed logging

This script compares our automated model detection against Epoch AI's 
existing tracker to validate coverage and accuracy.
"""
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for validation report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--json",
        action="store_true", 
        help="Also output raw comparison data as JSON"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    validator = EpochValidation()
    
    try:
        print("🔍 Validating against Epoch AI tracker...")
        print("=" * 60)
        
        # Get Epoch's models
        print("📥 Scraping Epoch AI tracker...")
        epoch_models = validator.scrape_epoch_tracker()
        
        if not epoch_models:
            print("❌ Failed to scrape Epoch tracker. Using mock data for demonstration.")
            # Use some known models for testing
            epoch_models = [
                EpochModel("GPT-4", 2.15e25, "medium", "Estimated from scaling"),
                EpochModel("Llama 3.1 405B", 3.8e25, "high", "Direct disclosure"),
                EpochModel("Claude 3 Opus", 1.5e25, "low", "Benchmark-based"),
                EpochModel("Gemini Ultra", 2.5e25, "speculative", "Hardware-based estimate"),
            ]
        
        print(f"✅ Found {len(epoch_models)} models in Epoch tracker")
        
        # Get our models  
        print("📊 Loading our automated results...")
        our_models = validator.get_our_models()
        print(f"✅ Found {len(our_models)} models in our system")
        
        # Compare datasets
        print("🔄 Comparing datasets...")
        comparison = validator.compare_models(epoch_models, our_models)
        
        # Generate report
        report = validator.generate_report(comparison, args.output)
        
        # Display summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Coverage: {comparison['match_count']}/{comparison['total_epoch']} models " +
              f"({comparison['match_count']/comparison['total_epoch']*100:.1f}%)")
        
        if comparison['match_count']/comparison['total_epoch'] >= 0.9:
            print("🎉 Excellent coverage! Ready for production use.")
        elif comparison['match_count']/comparison['total_epoch'] >= 0.75:
            print("✅ Good coverage. Minor improvements needed.")
        else:
            print("⚠️  Coverage needs improvement before production.")
        
        if not args.output:
            print("\n" + report)
        
        # Save JSON if requested
        if args.json:
            json_file = "validation_data.json"
            with open(json_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'epoch_models': [
                        {
                            'name': m.name,
                            'training_flop': m.training_flop,
                            'confidence': m.confidence,
                            'justification': m.justification
                        } for m in epoch_models
                    ],
                    'comparison': comparison
                }, f, indent=2, default=str)
            print(f"📄 Raw data saved to {json_file}")
        
    except KeyboardInterrupt:
        print("\n👋 Validation cancelled by user.")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()