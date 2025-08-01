#!/usr/bin/env python3
"""
Query tool for exploring model data.

This script provides an interface for querying, filtering, and analyzing
the model database with various output formats.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.query import ModelQueryEngine
from epoch_tracker.query.formatters import TableFormatter, FullFormatter
from epoch_tracker.models import Model


def format_flop(flop: Optional[float]) -> str:
    """Format FLOP value in scientific notation."""
    if flop is None:
        return "N/A"
    return f"{flop:.2e}"


def print_stats(models: List[Model]):
    """Print statistics about the models."""
    total = len(models)
    with_flops = len([m for m in models if m.training_flop])
    above_threshold = len([m for m in models if m.training_flop and m.training_flop >= 1e25])
    
    # Confidence breakdown
    confidence_counts = {"high": 0, "medium": 0, "low": 0, "speculative": 0}
    for model in models:
        if model.training_flop_confidence:
            confidence_counts[model.training_flop_confidence.value] += 1
    
    # Developer breakdown
    developer_counts = {}
    for model in models:
        dev = model.developer
        developer_counts[dev] = developer_counts.get(dev, 0) + 1
    
    print("=" * 60)
    print("MODEL DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total models: {total}")
    print(f"Models with FLOP estimates: {with_flops}")
    print(f"Models above 1e25 FLOP threshold: {above_threshold}")
    print()
    print("Confidence levels:")
    for level, count in confidence_counts.items():
        if count > 0:
            print(f"  {level}: {count}")
    print()
    print("Top developers by model count:")
    sorted_devs = sorted(developer_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for dev, count in sorted_devs:
        print(f"  {dev}: {count}")
    print("=" * 60)


def main():
    """Main entry point for query tool."""
    parser = argparse.ArgumentParser(
        description="Query and analyze model data"
    )
    
    # Filter options
    parser.add_argument(
        "--above-threshold",
        action="store_true",
        help="Show only models above 1e25 FLOP threshold"
    )
    parser.add_argument(
        "--developer",
        type=str,
        help="Filter by developer (case-insensitive)"
    )
    parser.add_argument(
        "--confidence",
        choices=["high", "medium", "low", "speculative"],
        help="Filter by confidence level"
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        help="Filter models whose name contains this string (case-insensitive)"
    )
    
    # Display options
    parser.add_argument(
        "--format",
        choices=["table", "detailed"],
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "developer", "flop", "release_date"],
        default="flop",
        help="Sort field (default: flop)"
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse sort order"
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Show only top N results"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics instead of model list"
    )
    
    # Data source options
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific data files to load (without .json extension)"
    )
    
    args = parser.parse_args()
    
    # Initialize query engine
    engine = ModelQueryEngine(args.data_dir)
    
    # Load data
    models = engine.load_data(args.files)
    
    if not models:
        print("No models found in data directory")
        sys.exit(1)
    
    # Apply filters
    if args.above_threshold:
        models = [m for m in models if m.training_flop and m.training_flop >= 1e25]
    
    if args.developer:
        dev_lower = args.developer.lower()
        models = [m for m in models if dev_lower in m.developer.lower()]
    
    if args.confidence:
        models = [m for m in models if m.training_flop_confidence and m.training_flop_confidence.value == args.confidence]
    
    if args.name_contains:
        name_lower = args.name_contains.lower()
        models = [m for m in models if name_lower in m.name.lower()]
    
    # Sort results
    if args.sort_by == "name":
        models.sort(key=lambda m: m.name.lower(), reverse=args.reverse)
    elif args.sort_by == "developer":
        models.sort(key=lambda m: m.developer.lower(), reverse=args.reverse)
    elif args.sort_by == "flop":
        models.sort(key=lambda m: m.training_flop or 0, reverse=not args.reverse)
    elif args.sort_by == "release_date":
        models.sort(key=lambda m: m.release_date or "", reverse=args.reverse)
    
    # Limit results
    if args.top:
        models = models[:args.top]
    
    # Display results
    if args.stats:
        print_stats(models)
    else:
        if args.format == "table":
            formatter = TableFormatter()
            print(formatter.format_models(models))
        elif args.format == "detailed":
            formatter = FullFormatter()
            print(formatter.format_models(models))
        
        print(f"\nTotal results: {len(models)}")


if __name__ == "__main__":
    main()