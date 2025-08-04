#!/usr/bin/env python3
"""
Interactive developer review CLI for managing the developer blacklist.

This script allows users to review new developers discovered during data collection
and make blacklist decisions with appropriate reasoning.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.utils.developer_blacklist import DeveloperBlacklist
from epoch_tracker.storage import JSONStorage


def review_developer_interactively(blacklist: DeveloperBlacklist, developer: str, 
                                  storage: JSONStorage) -> None:
    """Interactively review a single developer for blacklist decision.
    
    Args:
        blacklist: DeveloperBlacklist instance
        developer: Developer name to review
        storage: Storage instance for examining models
    """
    print(f"\n{'='*60}")
    print(f"REVIEWING DEVELOPER: {developer}")
    print(f"{'='*60}")
    
    # Show sample models from this developer
    try:
        all_models = storage.load_all_scraped_models()
        dev_models = [m for m in all_models if m.developer == developer]
        
        if dev_models:
            print(f"\nFound {len(dev_models)} models from {developer}:")
            for i, model in enumerate(dev_models[:10], 1):  # Show first 10
                benchmarks = ", ".join([f"{k}:{v}" for k, v in model.benchmarks.items()]) if model.benchmarks else "No benchmarks"
                print(f"  {i}. {model.name} - {benchmarks}")
            
            if len(dev_models) > 10:
                print(f"  ... and {len(dev_models) - 10} more models")
        else:
            print(f"\nNo models found from {developer} in current data")
    except Exception as e:
        print(f"\nCould not load models for review: {e}")
    
    # Get blacklist decision
    while True:
        print(f"\nShould {developer} be blacklisted? (y/n/skip): ", end="")
        decision = input().strip().lower()
        
        if decision in ['y', 'yes']:
            blacklisted = True
            break
        elif decision in ['n', 'no']:
            blacklisted = False
            break
        elif decision in ['s', 'skip']:
            print(f"Skipping {developer} - will remain in pending review")
            return
        else:
            print("Please enter 'y' for yes, 'n' for no, or 'skip' to skip this developer")
    
    # Get reason for the decision
    if blacklisted:
        print(f"\nReason for blacklisting {developer} (describe why they don't meet disclosure standards): ", end="")
        reason = input().strip()
        if not reason:
            reason = "Insufficient disclosure on training methodology and specifications"
    else:
        print(f"\nReason for allowing {developer} (describe their disclosure standards): ", end="")
        reason = input().strip()
        if not reason:
            reason = f"Adequate disclosure standards for model development - reviewed {datetime.now().strftime('%m.%d.%Y')}"
    
    # Update blacklist
    blacklist.update_developer_status(
        developer=developer,
        blacklisted=blacklisted,
        reason=reason
    )
    
    status_str = "BLACKLISTED" if blacklisted else "ALLOWED"
    print(f"\n✓ {developer} marked as {status_str}")


def show_blacklist_summary(blacklist: DeveloperBlacklist) -> None:
    """Show a summary of the current blacklist status."""
    stats = blacklist.get_statistics()
    blacklisted_devs = blacklist.get_blacklisted_developers()
    allowed_devs = blacklist.get_allowed_developers()
    pending_devs = blacklist.get_pending_review()
    
    print(f"\n{'='*60}")
    print("DEVELOPER BLACKLIST SUMMARY")
    print(f"{'='*60}")
    print(f"Total developers: {stats.get('total_developers', 0)}")
    print(f"Blacklisted: {stats.get('blacklisted_count', 0)}")
    print(f"Allowed: {stats.get('allowed_count', 0)}")
    print(f"Pending review: {stats.get('pending_count', 0)}")
    
    if blacklisted_devs:
        print(f"\nBlacklisted developers ({len(blacklisted_devs)}):")
        for dev in sorted(blacklisted_devs):
            reason = blacklist.get_blacklist_reason(dev)
            print(f"  • {dev} - {reason}")
    
    if pending_devs:
        print(f"\nPending review ({len(pending_devs)}):")
        for dev in sorted(pending_devs):
            print(f"  • {dev}")
    
    print(f"{'='*60}")


def main():
    """Main entry point for developer review CLI."""
    parser = argparse.ArgumentParser(
        description="Interactive developer blacklist review tool"
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Show current blacklist summary and exit"
    )
    parser.add_argument(
        "--developer",
        type=str,
        help="Review a specific developer (otherwise reviews all pending)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory for examining models (default: data)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize blacklist and storage
        blacklist = DeveloperBlacklist()
        storage = JSONStorage(args.data_dir)
        
        # Show summary if requested
        if args.show_summary:
            show_blacklist_summary(blacklist)
            return
        
        # Get developers to review
        if args.developer:
            developers_to_review = [args.developer]
        else:
            developers_to_review = blacklist.get_pending_review()
        
        if not developers_to_review:
            print("No developers need review!")
            show_blacklist_summary(blacklist)
            return
        
        print(f"Found {len(developers_to_review)} developers to review")
        
        # Review each developer
        reviewed_count = 0
        for developer in developers_to_review:
            try:
                review_developer_interactively(blacklist, developer, storage)
                reviewed_count += 1
            except KeyboardInterrupt:
                print(f"\n\nReview interrupted by user")
                break
            except Exception as e:
                print(f"\nError reviewing {developer}: {e}")
                continue
        
        # Save changes
        if reviewed_count > 0:
            blacklist.save_config()
            print(f"\n✓ Saved changes for {reviewed_count} developers")
        
        # Show final summary
        show_blacklist_summary(blacklist)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()