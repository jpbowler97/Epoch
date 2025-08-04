#!/usr/bin/env python3
"""
Standalone script to update complex benchmarks via Claude Code.

This script is designed to be run independently or via cron/scheduled tasks
to update JavaScript-heavy benchmark sites that regular scrapers cannot handle.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epoch_tracker.scrapers.claude_service import ClaudeScraperWithFallback


def setup_logging(verbose: bool = False):
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_manual_instructions(sites: list, config: dict) -> str:
    """Create detailed manual instructions for Claude Code execution."""
    instructions = []
    instructions.append("# Manual Benchmark Update Instructions for Claude Code")
    instructions.append("")
    instructions.append("## Task Overview")
    instructions.append("You need to fetch and save benchmark data from JavaScript-rendered websites.")
    instructions.append("Use your WebFetch tool to get the content, then save as HTML files.")
    instructions.append("")
    
    for i, site_name in enumerate(sites, 1):
        site_config = config["claude_scrapers"][site_name]
        instructions.append(f"## {i}. {site_name.upper()}")
        instructions.append("")
        instructions.append(f"**URL:** {site_config['url']}")
        instructions.append(f"**Output File:** {site_config['output_file']}")
        instructions.append("")
        instructions.append("**Extraction Requirements:**")
        for line in site_config['extraction_prompt'].split('\n'):
            if line.strip():
                instructions.append(f"  {line}")
        instructions.append("")
        instructions.append("**HTML Structure Required:**")
        instructions.append("```html")
        instructions.append("<!DOCTYPE html>")
        instructions.append("<html>")
        instructions.append("<head>")
        instructions.append("    <meta charset=\"UTF-8\">")
        instructions.append(f"    <title>{site_name} Benchmark Data</title>")
        instructions.append(f"    <!-- Updated: {datetime.now().isoformat()} -->")
        instructions.append("</head>")
        instructions.append("<body>")
        instructions.append("    <table>")
        instructions.append("        <thead><tr><!-- headers --></tr></thead>")
        instructions.append("        <tbody><!-- data rows --></tbody>")
        instructions.append("    </table>")
        instructions.append("</body>")
        instructions.append("</html>")
        instructions.append("```")
        instructions.append("")
    
    instructions.append("## Important Notes")
    instructions.append("- Use WebFetch to handle JavaScript rendering")
    instructions.append("- Preserve all model names and scores exactly")
    instructions.append("- Include timestamp comments in each file")
    instructions.append("- Create files even with partial data (log issues)")
    
    return "\n".join(instructions)


def main():
    """Main entry point for Claude benchmark updates."""
    parser = argparse.ArgumentParser(
        description="Update complex benchmark sites via Claude Code"
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        help="Specific sites to update (default: all that need updating)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update regardless of update frequency"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate instructions without executing"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Generate manual instructions for copy-paste execution"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--output-instructions",
        type=Path,
        help="Save instructions to file instead of executing"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Claude benchmark update script started")
    
    # Initialize Claude scraper
    try:
        claude_scraper = ClaudeScraperWithFallback()
    except Exception as e:
        logger.error(f"Failed to initialize Claude scraper: {e}")
        sys.exit(1)
    
    # Determine which sites to update
    if args.sites:
        # Validate specified sites
        valid_sites = []
        for site in args.sites:
            if site in claude_scraper.config["claude_scrapers"]:
                valid_sites.append(site)
            else:
                logger.warning(f"Unknown site: {site}")
        sites_to_update = valid_sites
    elif args.force:
        # Force update all sites
        sites_to_update = list(claude_scraper.config["claude_scrapers"].keys())
    else:
        # Update only sites that need it
        sites_to_update = claude_scraper.get_sites_needing_update()
    
    if not sites_to_update:
        logger.info("No sites need updating")
        print("All benchmark sites are up to date.")
        return 0
    
    logger.info(f"Sites to update: {sites_to_update}")
    
    # Generate instructions
    if args.manual:
        instructions = create_manual_instructions(sites_to_update, claude_scraper.config)
    else:
        instructions = claude_scraper.generate_instructions(sites_to_update)
    
    # Handle different execution modes
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Instructions that would be sent to Claude:")
        print("="*60)
        print(instructions)
        print("="*60)
        return 0
    
    if args.output_instructions:
        # Save instructions to file
        args.output_instructions.parent.mkdir(parents=True, exist_ok=True)
        args.output_instructions.write_text(instructions)
        logger.info(f"Instructions saved to {args.output_instructions}")
        print(f"Instructions saved to: {args.output_instructions}")
        print("\nTo execute manually:")
        print(f"  claude code --file {args.output_instructions}")
        return 0
    
    if args.manual:
        # Print manual instructions for copy-paste
        print("\n" + "="*60)
        print("MANUAL INSTRUCTIONS FOR CLAUDE CODE")
        print("="*60)
        print(instructions)
        print("="*60)
        print("\nCopy the above instructions and paste into Claude Code")
        return 0
    
    # Show manual instructions
    command = claude_scraper.show_manual_instructions(sites_to_update)
    
    if not args.dry_run and not args.manual and not args.output_instructions:
        print("\nWaiting for you to run the Claude command...")
        input("Press Enter after Claude has completed the updates...")
        
        # Mark as updated
        claude_scraper.mark_updated(sites_to_update)
        print("✓ Sites marked as updated!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())