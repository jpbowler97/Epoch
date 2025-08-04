# Scraper Architecture Documentation

## Overview

The Epoch tracker uses a modular, configuration-driven scraper architecture that supports both web-based and file-based data sources. This document describes the architecture, components, and how to extend the system.

## Architecture Components

### 1. Base Classes

#### `BaseScraper`
- Abstract base class for all scrapers
- Defines the common interface (`scrape_models()`)
- Provides utilities for data saving and logging

#### `WebScraperBase`
- Base class for HTTP-based scrapers
- Handles web requests, retries, and caching
- Uses `WebParser` for HTML table extraction

#### `FileScraperBase`
- Base class for local file scrapers
- Supports CSV, HTML, and Markdown files
- Handles file discovery and fallback patterns

### 2. Parser Modules

Located in `src/epoch_tracker/scrapers/parsers/`:

- **CSVParser**: Handles CSV file parsing with flexible column mapping
- **WebParser**: Extracts data from HTML tables via HTTP
- **FileParser**: Parses local HTML/XML files
- **MarkdownParser**: Extracts tables from Markdown documents

### 3. Utility Modules

Located in `src/epoch_tracker/scrapers/utils/`:

- **NumericParser**: Consistent numeric value parsing across all sources
- **ColumnMapper**: Flexible column name matching and mapping
- **ModelFactory**: Standardized model object creation with benchmark assignment

### 4. ConfigurableScraper

The unified scraper that reads JSON configurations and delegates to appropriate base classes.

## Configuration Format

Scrapers are defined using JSON configuration files in `configs/scrapers/`:

```json
{
  "name": "scraper_name",
  "type": "csv_file|web_table|html_file|markdown_file",
  "source": {
    "type": "http|local_files|directory",
    "url": "https://...",           // For web sources
    "paths": ["path/to/file.csv"],  // For file sources
    "fallback_pattern": "*.csv",    // Optional fallback
    "headers": {},                   // HTTP headers
    "timeout": 30                    // Request timeout
  },
  "parser": {
    "columns": {                     // Column mapping
      "model": "Model Name",
      "developer": "Organization",
      "score": "Score"
    },
    "benchmarks": {                  // Benchmark extraction
      "benchmark_name": "score_field"
    },
    "column_detection": {            // Alternative: auto-detect columns
      "model": ["model", "name"],
      "developer": ["org", "organization"]
    }
  },
  "metadata": {
    "description": "Scraper description",
    "source_name": "Data Source Name",
    "source_url": "https://..."
  }
}
```

## Data Flow

1. **Configuration Loading**: JSON config specifies source and parsing rules
2. **Data Acquisition**: 
   - Web scrapers fetch via HTTP
   - File scrapers read local files
3. **Parsing**: Raw data extracted using appropriate parser
4. **Mapping**: Columns mapped to standardized field names
5. **Model Creation**: ModelFactory creates Model objects with benchmarks
6. **Storage**: Models saved to JSON for further processing

## Adding a New Scraper

### Method 1: Configuration-Only (Recommended)

1. Create a JSON configuration file in `configs/scrapers/`
2. Define source, parser, and metadata sections
3. No code changes required!

Example for a new CSV source:
```json
{
  "name": "new_leaderboard",
  "type": "csv_file",
  "source": {
    "type": "local_files",
    "paths": ["data/raw/new_leaderboard.csv"]
  },
  "parser": {
    "columns": {
      "model": "Model",
      "developer": "Company",
      "score": "ELO Rating"
    },
    "benchmarks": {
      "new_leaderboard_elo": "score"
    }
  }
}
```

### Method 2: Custom Scraper Class

For complex sources requiring custom logic:

1. Extend `WebScraperBase` or `FileScraperBase`
2. Override `scrape_models()` method
3. Use utility classes for consistency
4. Register in scraper registry

## Benefits of New Architecture

### Code Reuse
- ~70% reduction in scraper code
- Single implementation for each data format
- Shared utilities across all scrapers

### Maintainability
- Configuration changes don't require code changes
- Consistent error handling and logging
- Centralized parsing logic

### Extensibility
- Add new scrapers with JSON only
- Support new formats by adding parsers
- Easy A/B testing of configurations

### Consistency
- Standardized model creation
- Uniform benchmark naming
- Consistent metadata structure

## Migration from Legacy Scrapers

Legacy scrapers can coexist with configurable scrapers. The `ScraperRegistry` manages both types transparently:

1. **Legacy scrapers**: Instantiated directly from Python classes
2. **Configurable scrapers**: Created from JSON configurations
3. **Unified interface**: Both types accessed through registry

## Testing

Use the test script to verify scraper configurations:

```bash
python scripts/test_configurable_scrapers.py
```

This compares output between legacy and configurable scrapers to ensure compatibility.

## Common Patterns

### Multiple File Sources with Fallback
```json
"source": {
  "type": "local_files",
  "paths": ["primary.csv"],
  "fallback_pattern": "backup_*.csv"
}
```

### Column Auto-Detection
```json
"column_detection": {
  "model": ["model", "name", "model_name"],
  "score": ["score", "elo", "rating"]
}
```

### Benchmark Mapping
```json
"benchmarks": {
  "benchmark_name": "source_field",
  "elo_rating": "elo",
  "accuracy": "acc"
}
```

## Troubleshooting

### No models extracted
- Check column mappings match actual column names
- Verify file paths are correct
- Check logs for parsing errors

### Benchmarks not appearing
- Ensure benchmark fields reference mapped column names
- Check numeric parsing for the benchmark values

### Web scraping failures
- Verify URL is accessible
- Check for required headers or authentication
- Review HTML structure for table selector