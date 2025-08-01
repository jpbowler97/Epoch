#!/usr/bin/env python3
"""
Main entry point for Epoch AI Model Tracker scripts.

This script provides easy access to all functionality in the reorganized scripts directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path: str, args: list = None) -> int:
    """Run a script with optional arguments.
    
    Args:
        script_path: Path to the script relative to scripts directory
        args: List of arguments to pass to the script
        
    Returns:
        Exit code from the script
    """
    script_full_path = Path(__file__).parent / script_path
    
    if not script_full_path.exists():
        print(f"Error: Script not found: {script_full_path}")
        return 1
    
    cmd = [sys.executable, str(script_full_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"Error running script: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Epoch AI Model Tracker - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:

Data Collection:
  collect-lmarena     Fetch LMArena leaderboard data
  collect-openlm      Fetch OpenLM arena data  
  collect-all         Collect data from all configured sources

Data Processing:
  estimate-flops      Apply FLOP estimations to scraped models
  refresh-dataset     Generate/refresh core dataset above 1e25 FLOP

Curation:
  manual-entry        Manual model entry interface
  review-candidates   Review candidate models for inclusion
  sync-datasets       Sync staging and published datasets

Analysis:
  query               Interactive data exploration and querying

Testing:
  test-estimator      Test compute estimation methods
  test-workflow       Test verification workflow
  validate            Validate against Epoch's existing data

Examples:
  %(prog)s collect-all                    # Full data collection pipeline
  %(prog)s estimate-flops --update        # Update FLOP estimates
  %(prog)s review-candidates               # Interactive model review
  %(prog)s query --above-threshold         # Query models above threshold
  %(prog)s sync-datasets --check-mappings # Check sync mappings
  %(prog)s sync-datasets --diff           # Interactive field value comparison
        """
    )
    
    parser.add_argument(
        'command',
        choices=[
            # Data collection
            'collect-lmarena', 'collect-openlm', 'collect-all',
            # Data processing  
            'estimate-flops', 'refresh-dataset',
            # Curation
            'manual-entry', 'review-candidates', 'sync-datasets',
            # Analysis
            'query',
            # Testing
            'test-estimator', 'test-workflow', 'validate'
        ],
        help='Command to execute'
    )
    
    # Parse known args to allow passing through unknown args to the target script
    args, unknown_args = parser.parse_known_args()
    
    # Map commands to script paths
    command_map = {
        # Data collection
        'collect-lmarena': 'data_collection/fetch_lmarena_data.py',
        'collect-openlm': 'data_collection/fetch_openlm_data.py', 
        'collect-all': 'data_collection/get_latest_model_data.py',
        
        # Data processing
        'estimate-flops': 'data_processing/estimate_flops.py',
        'refresh-dataset': 'data_processing/refresh_core_dataset.py',
        
        # Curation
        'manual-entry': 'curation/manual_model_entry.py',
        'review-candidates': 'curation/review_candidates.py',
        'sync-datasets': 'curation/sync_staging_published.py',
        
        # Analysis
        'query': 'analysis/query_models.py',
        
        # Testing
        'test-estimator': 'testing/test_compute_estimator.py', 
        'test-workflow': 'testing/test_verification_workflow.py',
        'validate': 'testing/validate_against_epoch.py',
    }
    
    script_path = command_map[args.command]
    exit_code = run_script(script_path, unknown_args)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()