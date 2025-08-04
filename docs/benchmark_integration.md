# Benchmark Integration Guide

This guide explains how to add new benchmark sources to the Epoch Model Tracker using the configurable benchmark scraper system.

## Overview

The benchmark scraper system allows you to integrate new AI model benchmark data without writing any code. Simply create a JSON configuration file that describes:

- Where the data is located (HTML files, CSV files, or Markdown tables)
- How to extract model names and scores from the data
- Which models to focus on
- Metadata about the benchmark source

## Quick Start

1. **Prepare your data file**: Save benchmark data as HTML, CSV, or Markdown
2. **Create configuration**: Copy `configs/benchmark_sources/template.json`
3. **Configure extraction**: Specify columns, filters, and parsing rules
4. **Run the scraper**: Use `get_latest_model_data.py --benchmark-config your_config.json`

## Configuration Format

### Basic Structure

```json
{
  "name": "Your Benchmark Collection",
  "description": "Brief description of what these benchmarks measure",
  "sources": [
    {
      "name": "benchmark_name",
      "type": "html_table|markdown_table|csv",
      "file_path": "path/to/your/benchmark/file",
      "parser_config": {
        // Parsing configuration here
      },
      "metadata": {
        // Additional information about the benchmark
      }
    }
  ]
}
```

### Supported File Types

#### HTML Tables (`"type": "html_table"`)

For HTML files containing `<table>` elements:

```json
{
  "name": "example_html_benchmark",
  "type": "html_table",
  "file_path": "data/benchmark_files/example.html",
  "parser_config": {
    "model_column": 0,
    "score_columns": {
      "accuracy": 1,
      "speed": 2
    },
    "filter_models": ["gpt", "claude"],
    "encoding": "utf-8",
    "headers": ["Model", "Accuracy", "Speed", "Notes"]
  }
}
```

#### CSV Files (`"type": "csv"`)

For comma-separated value files:

```json
{
  "name": "example_csv_benchmark", 
  "type": "csv",
  "file_path": "data/benchmark_files/example.csv",
  "parser_config": {
    "model_column": "Model Name",
    "score_columns": {
      "score": "Overall Score",
      "latency": "Response Time"
    },
    "filter_models": ["doubao"],
    "encoding": "utf-8",
    "delimiter": ","
  }
}
```

#### Markdown Tables (`"type": "markdown_table"`)

For GitHub-style markdown tables:

```json
{
  "name": "example_markdown_benchmark",
  "type": "markdown_table", 
  "file_path": "data/benchmark_files/example.md",
  "parser_config": {
    "model_column": 0,
    "score_columns": {
      "average": 1,
      "task_1": 2,
      "task_2": 3
    },
    "filter_models": ["doubao"],
    "encoding": "utf-8"
  }
}
```

## Parser Configuration Options

### Core Fields

- **`model_column`**: Column containing model names (integer index or string name)
- **`score_columns`**: Map of `score_name` → column reference for benchmark scores
- **`filter_models`**: List of strings - only extract rows containing these patterns
- **`encoding`**: File encoding (default: "utf-8")

### HTML-Specific Options

- **`headers`**: Explicit column names if table headers are missing/unclear

### CSV-Specific Options

- **`delimiter`**: Column separator character (default: ",")

### Column References

You can reference columns by:
- **Integer index**: `0, 1, 2, ...` (0-based indexing)
- **String name**: `"Model Name", "Accuracy", ...` (for CSV with headers)

## Model Filtering

Use `filter_models` to extract only relevant models:

```json
"filter_models": ["gpt", "claude", "gemini"]
```

This will only extract rows where the model name contains any of these patterns (case-insensitive).

## File Paths

File paths can be:
- **Relative to project root**: `"data/benchmark_files/example.html"`
- **Absolute paths**: `"/full/path/to/benchmark/file.csv"`

## Metadata

Add descriptive metadata to help document the benchmark:

```json
"metadata": {
  "description": "Benchmark measuring reasoning capabilities",
  "url": "https://example.com/benchmark",
  "metrics": ["Accuracy", "Response Time", "Cost"],
  "model_focus": "Large language models",
  "date_collected": "2025-01-01"
}
```

## Examples

### Example 1: Simple HTML Leaderboard

For an HTML file with a simple leaderboard table:

```html
<table>
  <tr><th>Rank</th><th>Model</th><th>Score</th></tr>
  <tr><td>1</td><td>GPT-4</td><td>95.2</td></tr>
  <tr><td>2</td><td>Claude-3</td><td>93.1</td></tr>
</table>
```

Configuration:
```json
{
  "name": "simple_leaderboard",
  "type": "html_table",
  "file_path": "data/benchmarks/leaderboard.html",
  "parser_config": {
    "model_column": 1,
    "score_columns": {
      "overall_score": 2
    },
    "filter_models": []
  }
}
```

### Example 2: Multi-Task CSV

For a CSV with multiple task scores:

```csv
Model,Overall,Math,Coding,Reasoning
GPT-4,89.5,92.1,87.3,89.0
Claude-3,87.2,89.8,85.1,86.7
```

Configuration:
```json
{
  "name": "multi_task_benchmark",
  "type": "csv", 
  "file_path": "data/benchmarks/tasks.csv",
  "parser_config": {
    "model_column": "Model",
    "score_columns": {
      "overall": "Overall",
      "math": "Math", 
      "coding": "Coding",
      "reasoning": "Reasoning"
    },
    "filter_models": ["gpt", "claude"]
  }
}
```

## Running the Scraper

### Command Line Usage

```bash
# Run specific benchmark configuration
python scripts/data_collection/get_latest_model_data.py \
  --scrapers benchmarks \
  --benchmark-config configs/benchmark_sources/chinese_benchmarks.json

# Run benchmarks along with other scrapers
python scripts/data_collection/get_latest_model_data.py \
  --scrapers lmarena benchmarks \
  --benchmark-config configs/benchmark_sources/video_benchmarks.json

# Run with verbose logging for debugging
python scripts/data_collection/get_latest_model_data.py \
  --scrapers benchmarks \
  --benchmark-config configs/benchmark_sources/agent_benchmarks.json \
  --verbose
```

### Integration with Pipeline

The benchmark scraper integrates with the existing data pipeline:

1. **Data Collection**: Benchmark scraper extracts models and scores
2. **Normalization**: Model names are normalized using existing rules
3. **Storage**: Data is saved to `data/scraped/benchmark_scraper_models.json`
4. **FLOP Estimation**: Benchmark scores can be used for FLOP estimation
5. **Query**: Results appear in standard model queries

## Output Format

The scraper produces standard `Model` objects with:

- **`name`**: Extracted model name
- **`developer`**: Auto-detected from model name or "Unknown"
- **`benchmarks`**: Dictionary of `{benchmark_name_score: value}`
- **`sources`**: Source URL and description
- **`metadata`**: Additional benchmark information

Example benchmark names in output:
- `olympic_arena_math: 87.5`
- `chinese_simpleqa_f_score: 0.92`
- `physics_iq_physics_iq_score: 8.2`

## Troubleshooting

### Common Issues

1. **File not found**: Check file_path is correct relative to project root
2. **No models extracted**: Verify filter_models patterns match your data
3. **Parsing errors**: Check column indices and encoding settings
4. **Invalid JSON**: Validate configuration file syntax

### Debugging Tips

1. **Enable verbose logging**: Use `--verbose` flag
2. **Check file encoding**: Try different encodings (utf-8, latin-1, etc.)
3. **Test column references**: Verify model_column and score_columns indices
4. **Simplify filters**: Start with empty filter_models to see all data

### Validation

The scraper validates configurations on startup and will report specific errors:

```
Source 0: Missing required field 'model_column'
Source 1: Invalid type 'xml_table'. Must be one of: ['html_table', 'markdown_table', 'csv']
```

## Best Practices

1. **Use descriptive names**: Make source names and score names clear
2. **Document metadata**: Include URLs, descriptions, and collection dates
3. **Test incrementally**: Start with simple configurations and add complexity
4. **Version control**: Keep benchmark data files and configs in version control
5. **Validate data**: Check extracted data makes sense before full pipeline runs

## Adding to Existing Configurations

You can add new sources to existing configuration files:

```json
{
  "name": "Extended Chinese Benchmarks",
  "sources": [
    // ... existing sources ...
    {
      "name": "new_benchmark",
      "type": "csv",
      "file_path": "data/benchmark_files/new_data.csv",
      // ... configuration ...
    }
  ]
}
```

This allows grouping related benchmarks together while keeping configurations manageable.