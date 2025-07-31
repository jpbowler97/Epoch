# Setup Guide

This guide covers the initial setup and configuration of the Epoch AI Model Tracker.

## Prerequisites

- **Python 3.9+** (tested with 3.9, 3.10, 3.11)
- **Git** for cloning the repository
- **Internet connection** for scraping data

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd epoch-tracker
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the minimal required dependencies:
- `requests` - HTTP client for web scraping
- `pydantic` + `pydantic-settings` - Data validation and configuration  
- `python-dateutil` - Date parsing utilities
- `huggingface-hub` - Hugging Face API access
- `pyyaml` - YAML configuration files
- `python-dotenv` - Environment variable loading
- `pandas` - CSV export functionality

## Configuration

### Environment Variables (Optional)

Create a `.env` file for optional configuration:

```bash
cp .env.example .env
```

### YAML Configuration

The system uses `configs/default.yaml` for main configuration:

```yaml
scraping:
  default_delay: 1.0
  max_requests_per_minute: 60
  
sources:
  huggingface:
    enabled: true
    delay: 0.5
```

You can modify these settings as needed.

## Verify Installation

### 1. Test Dependencies

```bash
python -c "import epoch_tracker; print('✅ Installation successful')"
```

### 2. Run Help Commands

```bash
# Test scraper
python scripts/scrape_huggingface.py --help

# Test query tool  
python scripts/query_models.py --help
```

### 3. Check Directory Structure

Verify the project structure:
```
epoch-tracker/
├── src/epoch_tracker/       # Main package
├── scripts/                 # CLI tools
├── configs/                 # Configuration files
├── data/                    # Data storage (created on first run)
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## First Run

### 1. Scrape Initial Data

```bash
python scripts/scrape_huggingface.py
```

Expected output:
```
2025-07-28 13:00:00 - scrapers.huggingface - INFO - Starting Hugging Face model scrape
2025-07-28 13:00:05 - scrapers.huggingface - INFO - Found 103 large models
2025-07-28 13:00:30 - scrapers.huggingface - INFO - Successfully scraped 39 models from Hugging Face

=== 9 Models Above 1e25 FLOP Threshold ===
Meta-Llama-3.1-405B-Instruct-AWQ-INT4: 2.46e+25 FLOP (medium)
...
```

### 2. Query the Data

```bash
# View statistics
python scripts/query_models.py --stats

# Show top models
python scripts/query_models.py --top 5
```

### 3. Check Data Files

```bash
ls data/processed/
# Should show: huggingface_test.json

ls data/raw/huggingface/
# Should show: scrape_YYYYMMDD_HHMMSS.json
```

## Troubleshooting

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'epoch_tracker'`

**Solution**: Make sure you're in the project directory and activated the virtual environment:
```bash
cd epoch-tracker
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Network Issues

**Error**: `HTTPError: 429 Too Many Requests`

**Solutions**:
1. Add Hugging Face token to `.env` file
2. Increase delays in `configs/default.yaml`:
   ```yaml
   sources:
     huggingface:
       delay: 2.0  # Increase from 0.5 to 2.0 seconds
   ```

### Permission Issues

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**: Check data directory permissions:
```bash
chmod -R 755 data/
```

### Dependency Conflicts

**Error**: Package version conflicts

**Solution**: Use a fresh virtual environment:
```bash
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Development Setup

### Additional Dependencies

For development work, install additional tools:

```bash
# Install development dependencies
pip install pytest pytest-cov responses black isort flake8

# Or install with extras
pip install -e ".[dev]"
```

### Pre-commit Hooks

Set up code formatting:

```bash
# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Check linting
flake8 src/ tests/ scripts/
```

### Running Tests

```bash
# Run tests (when test suite is available)
pytest tests/

# Run with coverage
pytest --cov=epoch_tracker tests/
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | None | HF API token for higher rate limits |
| `DEFAULT_DELAY` | 1.0 | Default delay between requests (seconds) |
| `HUGGINGFACE_DELAY` | 0.5 | Delay for HF API requests |
| `DATA_DIR` | ./data | Directory for data storage |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

### YAML Configuration

Key configuration sections in `configs/default.yaml`:

```yaml
scraping:
  default_delay: 1.0              # Default request delay
  max_requests_per_minute: 60     # Rate limiting
  user_agent: "Epoch AI Model Tracker 0.1.0"

sources:
  huggingface:
    enabled: true                 # Enable HF scraper
    delay: 0.5                   # HF-specific delay
    
storage:
  data_dir: "./data"             # Data storage location
  format: "json"                 # Storage format
  
logging:
  level: "INFO"                  # Log level
  file: "logs/epoch_tracker.log" # Log file location
```

## Next Steps

Once setup is complete:

1. **Read the [Usage Guide](usage.md)** for detailed usage instructions
2. **Run regular scrapes** to keep data current
3. **Explore query options** to analyze the data
4. **Check [Notes.md](../Notes.md)** for research findings and model details

For questions or issues, check the troubleshooting section above or review the project documentation.