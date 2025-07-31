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
    
    def interactive_entry(self) -> Optional[Model]:
        """Interactive prompts to gather model information."""
        print("\n" + "="*60)
        print("MANUAL MODEL ENTRY SYSTEM")
        print("="*60)
        print("Enter model information step by step.")
        print("Press Enter to skip optional fields.")
        print("Type 'quit' at any time to exit.\n")
        
        try:
            # Basic information
            name = self._prompt_required("Model name", "e.g., GPT-5, Claude 4, Llama 4")
            if name == 'quit':
                return None
                
            developer = self._prompt_required("Developer/Organization", "e.g., OpenAI, Anthropic, Meta")
            if developer == 'quit':
                return None
            
            # Create model with basic info
            model = Model(name=name, developer=developer)
            
            # Release date
            release_date = self._prompt_optional("Release date", "YYYY-MM-DD format, e.g., 2024-07-29")
            if release_date and release_date != 'quit':
                try:
                    model.release_date = datetime.strptime(release_date, "%Y-%m-%d")
                except ValueError:
                    print("⚠️  Invalid date format, skipping...")
            
            # Technical specifications
            params_str = self._prompt_optional("Parameter count", "e.g., 405000000000 for 405B")
            if params_str and params_str != 'quit':
                try:
                    model.parameters = self._parse_parameter_count(params_str)
                except ValueError:
                    print("⚠️  Invalid parameter format, skipping...")
            
            # Architecture
            architecture = self._prompt_optional("Architecture", "e.g., transformer, MoE")
            if architecture and architecture != 'quit':
                model.architecture = architecture
            
            # Context length
            context_str = self._prompt_optional("Context length", "e.g., 128000 for 128k tokens")
            if context_str and context_str != 'quit':
                try:
                    model.context_length = int(context_str)
                except ValueError:
                    print("⚠️  Invalid context length, skipping...")
            
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
    
    def _prompt_required(self, field_name: str, example: str) -> str:
        """Prompt for required field with validation."""
        while True:
            value = input(f"📝 {field_name} (required): ").strip()
            if value.lower() == 'quit':
                return 'quit'
            if value:
                return value
            print(f"❌ {field_name} is required. {example}")
    
    def _prompt_optional(self, field_name: str, example: str) -> str:
        """Prompt for optional field."""
        value = input(f"📝 {field_name} (optional): ").strip()
        if not value:
            return ""
        return value
    
    def _prompt_sources(self) -> list:
        """Prompt for source URLs."""
        sources = []
        print("\n📚 Sources (URLs for papers, blogs, announcements)")
        print("Enter one URL per line. Press Enter on empty line when done.")
        
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
        
        # Handle suffixes like 405b, 7b, 1.3b
        if 'b' in params_str:
            number = float(params_str.replace('b', '').strip())
            return int(number * 1_000_000_000)
        elif 'm' in params_str:
            number = float(params_str.replace('m', '').strip())
            return int(number * 1_000_000)
        else:
            return int(params_str)
    
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
        
        tokens_str = input("📊 Training tokens (e.g., 15e12 for 15T): ").strip()
        if not tokens_str:
            return
        
        try:
            tokens = float(tokens_str)
            result = self.estimator.estimate_from_scaling_laws(model.parameters, int(tokens))
            
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
            print("❌ Invalid tokens format")
    
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
            print(f"Confidence: {model.flop_confidence.value if model.flop_confidence else 'Unknown'}")
            print(f"Method: {model.estimation_method.value if model.estimation_method else 'Unknown'}")
            print(f"Status: {model.status.value if model.status else 'Unknown'}")
        
        if model.sources:
            print(f"Sources: {len(model.sources)} URLs")
    
    def save_model(self, model: Model) -> str:
        """Save model to manual_entries collection."""
        try:
            # Try to load existing manual entries
            try:
                collection = self.storage.load_models("manual_entries")
                if not collection:
                    collection = ModelCollection(models=[], source="manual_entry")
            except:
                collection = ModelCollection(models=[], source="manual_entry")
            
            # Add the new model
            collection.models.append(model)
            
            # Save back to storage
            self.storage.save_models(collection, "manual_entries")
            
            print(f"\n✅ Model '{model.name}' saved successfully!")
            print(f"📁 Saved to: data/processed/manual_entries.json")
            
            return "manual_entries"
            
        except Exception as e:
            print(f"\n❌ Failed to save model: {e}")
            raise
    
    def list_manual_entries(self):
        """List all manually entered models."""
        try:
            collection = self.storage.load_models("manual_entries")
            if not collection or not collection.models:
                print("No manually entered models found.")
                return
            
            print(f"\n📋 Manual Entries ({len(collection.models)} models):")
            print("-" * 60)
            
            for i, model in enumerate(collection.models, 1):
                flop_str = f"{model.training_flop:.2e}" if model.training_flop else "Unknown"
                status_str = model.status.value if model.status else "Unknown"
                print(f"{i:2d}. {model.name} ({model.developer})")
                print(f"    FLOP: {flop_str} | Status: {status_str}")
            
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
                print("  python scripts/query_models.py --stats")
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