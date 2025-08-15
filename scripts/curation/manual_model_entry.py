#!/usr/bin/env python3
"""Manual model entry system for contractors to add/correct models."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.models import Model, ModelCollection, ConfidenceLevel, EstimationMethod, ModelStatus
from epoch_tracker.storage import JSONStorage
from epoch_tracker.estimation import ComputeEstimator


class ManualModelEntry:
    """Interactive system for manual model entry and correction."""
    
    def __init__(self):
        self.storage = JSONStorage()
        self.estimator = ComputeEstimator()
        self.logger = logging.getLogger(__name__)
        self.field_config = self._load_field_config()
    
    def _load_field_config(self) -> Dict[str, Any]:
        """Load field definitions from JSON config."""
        import json
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent.parent / "configs" / "manual_entry_fields.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load field config: {e}")
            # Return minimal fallback config
            return {}
    
    def _display_field_prompt(self, field_key: str) -> None:
        """Display consistent field prompt with definition and examples."""
        if field_key not in self.field_config:
            return
            
        config = self.field_config[field_key]
        
        # Header with icon, label, and required/optional
        required_text = "(required)" if config.get("required", False) else "(optional)"
        icon = config.get("icon", "📝")
        print(f"\n{icon} {config['label']} {required_text}:")
        
        # Definition
        if "definition" in config:
            print(f"   Definition: {config['definition']}")
        
        # Examples  
        if "examples" in config:
            examples_str = ", ".join(config['examples'])
            print(f"   Examples: {examples_str}")
        
        # Special notes
        if "note" in config:
            print(f"   Note: {config['note']}")
        
        # Special instructions (for sources, etc.)
        if "instruction" in config:
            print(f"   {config['instruction']}")
        
        # Format information (for dates, etc.)
        if "formats" in config:
            formats_str = " or ".join(config['formats'])
            print(f"   Formats: {formats_str}")
    
    def _check_duplicate_model(self, model_name: str) -> tuple[bool, str]:
        """Check if model already exists using normalized names.
        
        Returns:
            (is_duplicate, duplicate_info) - info about existing model if duplicate found
        """
        try:
            from epoch_tracker.utils.model_names import normalize_model_name
            import csv
            
            normalized_name = normalize_model_name(model_name)
            staging_csv_path = Path("data/staging/above_1e25_flop_staging.csv")
            
            if not staging_csv_path.exists():
                return False, ""
            
            with open(staging_csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_name = row.get('model', '')
                    if normalize_model_name(existing_name) == normalized_name:
                        # Found duplicate, create info string
                        developer = row.get('developer', 'Unknown')
                        release_date = row.get('release_date', 'Unknown')
                        params = row.get('parameters', '')
                        if params:
                            params_b = float(params) / 1_000_000_000 if params.replace('.','').isdigit() else params
                            params_str = f"{params_b:.0f}B parameters" if isinstance(params_b, float) else params
                        else:
                            params_str = "Unknown parameters"
                        
                        duplicate_info = f"{developer}, {release_date}, {params_str}"
                        return True, duplicate_info
                        
            return False, ""
            
        except Exception as e:
            self.logger.warning(f"Failed to check for duplicates: {e}")
            return False, ""
    
    def _prompt_field(self, field_key: str, required: bool = False) -> str:
        """Single method to handle all field prompts with clean display."""
        self._display_field_prompt(field_key)
        
        field_config = self.field_config[field_key]
        label = field_config['label']
        
        while True:
            if required:
                value = input(f"{label}: ").strip()
                if value.lower() == 'quit':
                    return 'quit'
                if value:
                    return value
                print(f"❌ {label} is required.")
            else:
                value = input(f"{label}: ").strip()
                return value if value.lower() != 'quit' else 'quit'
    
    def interactive_entry(self) -> Optional[Model]:
        """Interactive prompts to gather model information."""
        print("\n" + "="*60)
        print("MANUAL MODEL ENTRY SYSTEM")
        print("="*60)
        print("Enter model information step by step.")
        print("Press Enter to skip optional fields.")
        print("Type 'quit' at any time to exit.\n")
        
        try:
            # Basic information with duplicate checking
            name = self._prompt_field("model_name", required=True)
            if name == 'quit':
                return None
            
            # Check for duplicates
            is_duplicate, duplicate_info = self._check_duplicate_model(name)
            if is_duplicate:
                from epoch_tracker.utils.model_names import normalize_model_name
                normalized = normalize_model_name(name)
                print(f"\n⚠️  Model \"{normalized}\" already exists in staging dataset!")
                print(f"   Existing entry: {duplicate_info}")
                
                continue_anyway = input("   Continue anyway? (y/n): ").lower().strip()
                if continue_anyway not in ['y', 'yes']:
                    print("Model entry cancelled due to duplicate.")
                    return None
                print()  # Extra spacing after duplicate warning
                
            developer = self._prompt_field("developer", required=True)
            if developer == 'quit':
                return None
            
            # Create model with basic info
            model = Model(name=name, developer=developer)
            
            # Release date
            release_date = self._prompt_field("release_date", required=False)
            if release_date and release_date != 'quit':
                parsed_date = self._parse_date_flexible(release_date)
                if parsed_date:
                    model.release_date = parsed_date
                else:
                    print("⚠️  Invalid date format. Please use YYYY-MM-DD or MM/DD/YYYY (e.g., 2024-03-04 or 3/4/2024)")
            
            # Technical specifications
            params_str = self._prompt_field("parameters", required=False)
            if params_str and params_str != 'quit':
                try:
                    model.parameters = self._parse_parameter_count(params_str)
                    model.parameter_source = "manual_entry"
                except ValueError:
                    print("⚠️  Invalid parameter format. Use formats like: 405B, 70B, 1.76T, or raw numbers")
            
            # Architecture
            architecture = self._prompt_field("architecture", required=False)
            if architecture and architecture != 'quit':
                model.architecture = architecture
            
            # Context length
            context_str = self._prompt_field("context_length", required=False)
            if context_str and context_str != 'quit':
                try:
                    model.context_length = int(context_str)
                except ValueError:
                    print("⚠️  Invalid context length. Please enter a number (e.g., 128000 for 128k tokens)")
            
            # Sources
            sources = self._prompt_sources()
            if sources:
                model.sources = sources
            
            # FLOP estimation
            self._estimate_flop_interactive(model)
            
            # Final review
            print("\n" + "="*60)
            print("MODEL SUMMARY")
            print("="*60)
            self._print_model_summary(model)
            
            confirm = input("\nSave this model? (y/n): ").lower().strip()
            if confirm in ['y', 'yes']:
                return model
            else:
                print("Model entry cancelled.")
                return None
                
        except KeyboardInterrupt:
            print("\n\nModel entry cancelled by user.")
            return None
    
    
    def _prompt_sources(self) -> list:
        """Prompt for source URLs using centralized config."""
        sources = []
        self._display_field_prompt("sources")
        
        while True:
            url = input(f"Source {len(sources) + 1}: ").strip()
            if not url:
                break
            if url.lower() == 'quit':
                return []
            sources.append(url)
        
        return sources
    
    def _parse_parameter_count(self, params_str: str) -> int:
        """Parse parameter count from various formats."""
        params_str = params_str.lower().replace(',', '').replace('_', '')
        
        # Handle suffixes like 405b, 7b, 1.3b, 1.76t
        if 't' in params_str:
            number = float(params_str.replace('t', '').strip())
            return int(number * 1_000_000_000_000)
        elif 'b' in params_str:
            number = float(params_str.replace('b', '').strip())
            return int(number * 1_000_000_000)
        elif 'm' in params_str:
            number = float(params_str.replace('m', '').strip())
            return int(number * 1_000_000)
        else:
            return int(params_str)
    
    def _parse_token_count(self, tokens_str: str) -> int:
        """Parse token count from various formats."""
        tokens_str = tokens_str.lower().replace(',', '').replace('_', '')
        
        # Handle suffixes like 7t, 15t, 1.5t (trillion)
        if 't' in tokens_str:
            number = float(tokens_str.replace('t', '').strip())
            return int(number * 1_000_000_000_000)
        elif 'b' in tokens_str:
            number = float(tokens_str.replace('b', '').strip())
            return int(number * 1_000_000_000)
        elif 'm' in tokens_str:
            number = float(tokens_str.replace('m', '').strip())
            return int(number * 1_000_000)
        elif 'e' in tokens_str:
            # Scientific notation like 1.5e13
            return int(float(tokens_str))
        else:
            return int(float(tokens_str))
    
    def _parse_date_flexible(self, date_str: str) -> Optional[datetime]:
        """Parse dates in ISO or US format only."""
        formats = [
            '%Y-%m-%d',     # 2024-03-04 (ISO format)
            '%m/%d/%Y',     # 3/4/2024 or 03/04/2024 (US format)
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None
    
    def _estimate_flop_interactive(self, model: Model):
        """Interactive FLOP estimation."""
        print("\n" + "-"*40)
        print("COMPUTE ESTIMATION")
        print("-"*40)
        print("Choose how to estimate training compute:")
        print("1. Direct FLOP disclosure (most accurate)")
        print("2. Parameters + training tokens (scaling laws)")
        print("3. Hardware specifications")
        print("4. Training cost")
        print("5. Benchmark ELO rating")
        print("6. Skip (add manually later)")
        
        choice = input("Choose method (1-6): ").strip()
        
        if choice == '1':
            self._estimate_direct_flop(model)
        elif choice == '2':
            self._estimate_scaling_laws(model)
        elif choice == '3':
            self._estimate_hardware(model)
        elif choice == '4':
            self._estimate_cost(model)
        elif choice == '5':
            self._estimate_benchmark(model)
        else:
            print("Skipping FLOP estimation.")
    
    def _estimate_direct_flop(self, model: Model):
        """Direct FLOP disclosure entry."""
        flop_str = input("💻 Training FLOP (e.g., 3.8e25): ").strip()
        if not flop_str:
            return
        
        try:
            flop_value = float(flop_str)
            confidence = ConfidenceLevel.HIGH
            reasoning = f"Direct FLOP disclosure: {flop_value:.2e} FLOP"
            
            model.update_flop_estimate(
                flop=flop_value,
                confidence=confidence,
                method=EstimationMethod.COMPANY_DISCLOSURE,
                reasoning=reasoning
            )
            
            # Set status
            if flop_value >= 1e25:
                model.status = ModelStatus.CONFIRMED_ABOVE
            else:
                model.status = ModelStatus.CONFIRMED_BELOW
                
            print(f"✅ FLOP estimate set: {flop_value:.2e}")
            
        except ValueError:
            print("❌ Invalid FLOP format, skipping...")
    
    def _estimate_scaling_laws(self, model: Model):
        """Scaling laws estimation."""
        if not model.parameters:
            params_str = input("🔢 Parameters (required for scaling laws): ").strip()
            if params_str:
                try:
                    model.parameters = self._parse_parameter_count(params_str)
                except ValueError:
                    print("❌ Invalid parameter format")
                    return
            else:
                return
        
        print("\n📊 Training tokens options:")
        print("  • Press Enter: Use automatic era-aware estimation")
        print("  • Enter number: e.g., '7T' (7 trillion), '15T', or '1.5e13'")
        tokens_str = input("Training tokens: ").strip()
        
        if not tokens_str:
            # Use automatic era-aware token estimation
            print("Using automatic era-aware token estimation...")
            try:
                # Import the era-aware estimation function
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent / "data_processing"))
                from estimate_flops import estimate_training_tokens
                
                estimated_tokens, confidence, reasoning = estimate_training_tokens(model.name, model.parameters)
                print(f"📊 Estimated tokens: {estimated_tokens/1e12:.1f}T {reasoning}")
                
                result = self.estimator.estimate_from_scaling_laws(model.parameters, int(estimated_tokens))
                
                model.update_flop_estimate(
                    flop=result.flop_estimate,
                    confidence=result.confidence,
                    method=result.method,
                    reasoning=f"Era-aware estimation: {result.reasoning}"
                )
                
                # Set status
                if result.flop_estimate >= 1e25:
                    model.status = ModelStatus.LIKELY_ABOVE
                else:
                    model.status = ModelStatus.LIKELY_BELOW
                    
                print(f"✅ FLOP estimate: {result.flop_estimate:.2e} ({result.confidence.value})")
                
            except Exception as e:
                print(f"❌ Automatic estimation failed: {e}")
                return
        else:
            try:
                tokens = self._parse_token_count(tokens_str)
                print(f"📊 Using {tokens/1e12:.1f}T tokens")
                result = self.estimator.estimate_from_scaling_laws(model.parameters, tokens)
                
                model.update_flop_estimate(
                    flop=result.flop_estimate,
                    confidence=result.confidence,
                    method=result.method,
                    reasoning=result.reasoning
                )
                
                # Set status
                if result.flop_estimate >= 1e25:
                    model.status = ModelStatus.LIKELY_ABOVE
                else:
                    model.status = ModelStatus.LIKELY_BELOW
                    
                print(f"✅ FLOP estimate: {result.flop_estimate:.2e} ({result.confidence.value})")
                
            except ValueError as e:
                print(f"❌ Invalid tokens format. Please use formats like: 7T, 15T, 1.5e13")
                print(f"   Error: {e}")
                return
    
    def _estimate_hardware(self, model: Model):
        """Hardware-based estimation."""
        gpu_type = input("🖥️  GPU type (e.g., H100, A100): ").strip()
        if not gpu_type:
            return
        
        gpu_count_str = input("🔢 GPU count (e.g., 16000): ").strip()
        if not gpu_count_str:
            return
        
        training_hours_str = input("⏱️  Training hours (e.g., 2000): ").strip()
        if not training_hours_str:
            return
        
        try:
            gpu_count = int(gpu_count_str)
            training_hours = float(training_hours_str)
            
            result = self.estimator.estimate_from_hardware(
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                training_time_hours=training_hours
            )
            
            model.update_flop_estimate(
                flop=result.flop_estimate,
                confidence=result.confidence,
                method=result.method,
                reasoning=result.reasoning
            )
            
            # Set status
            if result.flop_estimate >= 1e25:
                model.status = ModelStatus.LIKELY_ABOVE
            else:
                model.status = ModelStatus.LIKELY_BELOW
            
            print(f"✅ FLOP estimate: {result.flop_estimate:.2e} ({result.confidence.value})")
            
        except ValueError:
            print("❌ Invalid hardware specifications")
    
    def _estimate_cost(self, model: Model):
        """Cost-based estimation."""
        cost_str = input("💰 Training cost in USD (e.g., 78000000): ").strip()
        if not cost_str:
            return
        
        gpu_type = input("🖥️  GPU type used (e.g., H100): ").strip() or "H100"
        
        try:
            cost = float(cost_str)
            result = self.estimator.estimate_from_cost_disclosure(cost, gpu_type)
            
            model.update_flop_estimate(
                flop=result.flop_estimate,
                confidence=result.confidence,
                method=result.method,
                reasoning=result.reasoning
            )
            
            # Set status based on estimate
            if result.flop_estimate >= 1e25:
                model.status = ModelStatus.LIKELY_ABOVE
            else:
                model.status = ModelStatus.LIKELY_BELOW
            
            print(f"✅ FLOP estimate: {result.flop_estimate:.2e} ({result.confidence.value})")
            
        except ValueError:
            print("❌ Invalid cost format")
    
    def _estimate_benchmark(self, model: Model):
        """Benchmark-based estimation."""
        elo_str = input("🏆 ChatBot Arena ELO rating (e.g., 1350): ").strip()
        if not elo_str:
            return
        
        try:
            elo = float(elo_str)
            result = self.estimator.estimate_from_benchmark_elo(elo)
            
            model.update_flop_estimate(
                flop=result.flop_estimate,
                confidence=result.confidence,
                method=result.method,
                reasoning=result.reasoning
            )
            
            # Set status
            if result.flop_estimate >= 1e25:
                model.status = ModelStatus.LIKELY_ABOVE
            else:
                model.status = ModelStatus.LIKELY_BELOW
            
            print(f"✅ FLOP estimate: {result.flop_estimate:.2e} ({result.confidence.value})")
            
        except ValueError:
            print("❌ Invalid ELO format")
    
    def _print_model_summary(self, model: Model):
        """Print a formatted model summary."""
        print(f"Name: {model.name}")
        print(f"Developer: {model.developer}")
        
        if model.release_date:
            print(f"Release Date: {model.release_date.strftime('%Y-%m-%d')}")
        
        if model.parameters:
            params_b = model.parameters / 1_000_000_000
            print(f"Parameters: {params_b:.1f}B")
        
        if model.architecture:
            print(f"Architecture: {model.architecture}")
        
        if model.context_length:
            print(f"Context Length: {model.context_length:,}")
        
        if model.training_flop:
            print(f"Training FLOP: {model.training_flop:.2e}")
            print(f"Above 1e25 threshold: {'✅ Yes' if model.training_flop >= 1e25 else '❌ No'}")
            print(f"Confidence: {model.training_flop_confidence.value if model.training_flop_confidence else 'Unknown'}")
            print(f"Method: {model.estimation_method.value if model.estimation_method else 'Unknown'}")
            print(f"Status: {model.status.value if model.status else 'Unknown'}")
        
        if model.sources:
            print(f"Sources: {len(model.sources)} URLs")
    
    def save_model(self, model: Model) -> str:
        """Save model directly to staging CSV dataset."""
        import csv
        from datetime import datetime
        
        try:
            staging_csv_path = Path("data/staging/above_1e25_flop_staging.csv")
            
            # Set verification flag for automation protection
            model.metadata = model.metadata or {}
            
            # Add lineage documentation to notes
            entry_date = datetime.now().strftime("%Y-%m-%d")
            lineage_note = f"Added via manual-entry on {entry_date} using scripts/run.py manual-entry"
            
            # Prepare row data matching CSV schema (19 fields)
            row_data = {
                'model': model.name,
                'developer': model.developer,
                'release_date': model.release_date.strftime("%Y-%m-%d") if model.release_date else '',
                'parameters': str(model.parameters) if model.parameters else '',
                'parameter_source': model.parameter_source or '',
                'training_flop': f"{model.training_flop:.2e}" if model.training_flop else '',
                'confidence': model.training_flop_confidence.value if model.training_flop_confidence else 'speculative',
                'confidence_explanation': model.reasoning or '',
                'estimation_method': model.estimation_method.value if model.estimation_method else 'manual_research',
                'alternative_methods': '',  # Not used in manual entry
                'threshold_classification': self._get_threshold_classification(model),
                'status': model.status.value if model.status else 'uncertain',
                'reasoning': model.reasoning or '',
                'sources': '; '.join(model.sources) if model.sources else '',
                'verified': 'y',  # Always set for manual entries
                'last_updated': model.last_updated.isoformat(),
                'notes': lineage_note,
                'blacklist_status': '',  # Not applicable for manual entries
                'original_estimate': f"{model.training_flop:.2e}" if model.training_flop else ''
            }
            
            # Check if file exists to determine if we need headers
            file_exists = staging_csv_path.exists()
            
            # Append to CSV
            with open(staging_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'model', 'developer', 'release_date', 'parameters', 'parameter_source',
                    'training_flop', 'confidence', 'confidence_explanation', 'estimation_method',
                    'alternative_methods', 'threshold_classification', 'status', 'reasoning',
                    'sources', 'verified', 'last_updated', 'notes', 'blacklist_status', 'original_estimate'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(row_data)
            
            print(f"\n✅ Model '{model.name}' saved successfully!")
            print(f"📁 Appended to: data/staging/above_1e25_flop_staging.csv")
            print(f"🔒 Protected with verified=y flag")
            
            return "staging"
            
        except Exception as e:
            print(f"\n❌ Failed to save model: {e}")
            raise
    
    def _get_threshold_classification(self, model: Model) -> str:
        """Get threshold classification based on FLOP estimate and confidence."""
        if not model.training_flop:
            return "uncertain"
        
        if model.training_flop >= 1e25:
            if hasattr(model.training_flop_confidence, 'value') and model.training_flop_confidence.value == 'high':
                return "high_confidence_above_1e25"
            elif model.training_flop_confidence == ConfidenceLevel.HIGH:
                return "high_confidence_above_1e25"
            else:
                return "likely_above_1e25"
        else:
            if hasattr(model.training_flop_confidence, 'value') and model.training_flop_confidence.value == 'high':
                return "high_confidence_below_1e25"
            elif model.training_flop_confidence == ConfidenceLevel.HIGH:
                return "high_confidence_below_1e25"
            else:
                return "likely_below_1e25"
    
    def list_manual_entries(self):
        """List all manually entered models from staging CSV."""
        import csv
        
        try:
            staging_csv_path = Path("data/staging/above_1e25_flop_staging.csv")
            
            if not staging_csv_path.exists():
                print("No staging dataset found.")
                return
            
            manual_entries = []
            with open(staging_csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if entry has manual-entry lineage in notes
                    if row['notes'] and 'manual-entry' in row['notes']:
                        manual_entries.append(row)
            
            if not manual_entries:
                print("No manually entered models found in staging dataset.")
                return
            
            print(f"\n📋 Manual Entries ({len(manual_entries)} models):")
            print("-" * 80)
            
            for i, entry in enumerate(manual_entries, 1):
                flop_str = entry['training_flop'] if entry['training_flop'] else "Unknown"
                status_str = entry['status'] or "Unknown"
                verified_str = "✅ Protected" if entry['verified'] == 'y' else "⚠️ Unprotected"
                print(f"{i:2d}. {entry['model']} ({entry['developer']})")
                print(f"    FLOP: {flop_str} | Status: {status_str} | {verified_str}")
                if entry['notes']:
                    print(f"    Notes: {entry['notes'][:60]}...")
            
        except Exception as e:
            print(f"❌ Failed to list manual entries: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manual model entry system for contractors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Interactive model entry
  %(prog)s --list            # List all manual entries
  %(prog)s --help            # Show this help message

This tool provides a contractor-friendly interface for manually adding
or correcting AI model information in the Epoch tracker.
"""
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all manually entered models"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    entry_system = ManualModelEntry()
    
    try:
        if args.list:
            entry_system.list_manual_entries()
        else:
            # Interactive entry mode
            model = entry_system.interactive_entry()
            if model:
                entry_system.save_model(model)
                print("\n🎉 Model entry completed successfully!")
                print("\nTo view all models including manual entries:")
                print("  python scripts/run.py query --stats")
            else:
                print("\n👋 Model entry cancelled.")
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()