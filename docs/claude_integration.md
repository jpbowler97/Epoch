# Claude Code Integration for Complex Web Scraping

## Overview

The Epoch tracker integrates with Claude Code's WebFetch capabilities to handle browser-interactive websites that regular Python scrapers cannot access. This hybrid approach allows us to collect data from modern web applications that require actual browser interaction (clicking, scrolling, waiting for dynamic content).

## Architecture

```
Regular Scrapers (Python)          Claude Code Service
├── HuggingFace API               ├── SuperCLUE (Gradio)
├── LMArena (Static HTML)          ├── Physics-IQ (Interactive)
├── OpenLM Arena (JSON)            ├── Olympic Arena (React)
└── Papers with Code               └── Video Arena (Dynamic)
                ↓                              ↓
        data/scraped/*.json          data/benchmark_files/*.html
                └──────────┬──────────┘
                           ↓
                   Unified Pipeline
```

## Components

### 1. Claude Service Module
- **Location**: `src/epoch_tracker/scrapers/claude_service.py`
- **Purpose**: Manages Claude Code execution for web scraping
- **Features**:
  - Automatic update scheduling based on frequency
  - Instruction generation from configuration
  - Subprocess execution with Claude CLI
  - Output validation and error handling
  - Fallback to cached data on failure

### 2. Configuration
- **Location**: `configs/claude_scraping.yaml`
- **Purpose**: Defines which sites require Claude's capabilities
- **Structure**:
  ```yaml
  claude_scrapers:
    site_name:
      url: "https://example.com"
      output_file: "data/benchmark_files/Example.html"
      extraction_prompt: "What to extract..."
      update_frequency: "daily|weekly|monthly"
  ```

### 3. Standalone Script
- **Location**: `scripts/update_claude_benchmarks.py`
- **Purpose**: Update complex sites independently
- **Usage**:
  ```bash
  # Update all sites needing refresh
  python scripts/update_claude_benchmarks.py
  
  # Force update specific sites
  python scripts/update_claude_benchmarks.py --sites superclue physics_iq --force
  
  # Generate instructions without executing
  python scripts/update_claude_benchmarks.py --dry-run
  
  # Generate manual instructions for copy-paste
  python scripts/update_claude_benchmarks.py --manual
  ```

### 4. Pipeline Integration
- **Location**: `scripts/data_collection/get_latest_model_data.py`
- **Flags**:
  - `--update-claude-sites`: Update sites that need refresh
  - `--force-claude-update`: Force update all Claude-managed sites
- **Example**:
  ```bash
  # Regular data collection with Claude updates
  python scripts/run.py collect-all --update-claude-sites
  ```

## How It Works

### The Problem
Many modern benchmark sites require interactive browser behavior to access their content:
- **SuperCLUE**: Uses Gradio interface requiring clicks to load data
- **Physics-IQ**: Interactive tables requiring navigation
- **Video Arena**: Dynamic content loading requiring scrolling and waiting

Regular Python scrapers using `requests` and `BeautifulSoup` cannot perform these browser interactions, so they miss this content.

### The Solution
Claude Code has access to WebFetch tools that can:
1. **Interact with Browser**: Full browser-like interaction (clicking, scrolling, waiting)
2. **Extract Content**: Parse content after all interactions are complete
3. **Save Locally**: Create HTML files for regular scrapers

### Simplified Workflow

1. **User runs update command**:
   ```bash
   python scripts/run.py update-claude
   ```

2. **System shows simple command**:
   ```
   claude code scripts/claude_instructions/update_benchmarks.md
   ```

3. **User runs the Claude command**:
   - Opens Claude Code with detailed instructions
   - Claude executes WebFetch commands automatically
   - Fresh HTML files saved to data/benchmark_files/

4. **User presses Enter**:
   - System marks sites as updated
   - Ready for regular data collection

5. **Automatic Warning System**:
   - If you run `collect-all` with stale Claude data (>24h)
   - Clear warning with instructions to update first

## Configuration Examples

### SuperCLUE Configuration
```yaml
superclue:
  url: "https://www.superclueai.com/"
  output_file: "data/benchmark_files/SuperCLUE.html"
  extraction_prompt: |
    Extract the complete leaderboard table with:
    - Model names (模型名称)
    - Organization (机构)
    - Overall score (总分)
    - Math, reasoning, code, agents scores
  update_frequency: "daily"
```

### Scraper Configuration
```json
{
  "name": "superclue",
  "type": "html_file",
  "source": {
    "type": "local_file",
    "path": "data/benchmark_files/SuperCLUE.html"
  },
  "parser": {
    "table_selector": "table",
    "columns": {
      "model": "Model",
      "organization": "Organization"
    }
  }
}
```

## Scheduling

### Manual Updates
```bash
# One-time update
python scripts/update_claude_benchmarks.py
```

### Cron Job
```cron
# Update daily at 2 AM
0 2 * * * cd /path/to/epoch && python scripts/update_claude_benchmarks.py
```

### GitHub Actions
```yaml
name: Update Claude Benchmarks
on:
  schedule:
    - cron: '0 2 * * *'
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Update benchmarks
        run: python scripts/update_claude_benchmarks.py
      - name: Commit changes
        run: |
          git add data/benchmark_files/
          git commit -m "Update benchmark data via Claude"
          git push
```

## Error Handling

The system includes multiple layers of error handling:

1. **Retry Logic**: Automatic retries on temporary failures
2. **Fallback to Cache**: Use existing files if update fails
3. **Partial Success**: Continue with working scrapers if some fail
4. **Validation**: Verify output before using
5. **Logging**: Detailed logs for debugging

## Benefits

1. **Comprehensive Coverage**: Access browser-interactive sites
2. **Automation**: Scheduled updates without manual intervention
3. **Reliability**: Fallback mechanisms ensure continuity
4. **Maintainability**: Clear separation of concerns
5. **Scalability**: Easy to add new complex sites

## Adding New Sites

To add a new browser-interactive site:

1. **Add to Configuration** (`configs/claude_scraping.yaml`):
   ```yaml
   new_site:
     url: "https://newsite.com"
     output_file: "data/benchmark_files/NewSite.html"
     extraction_prompt: "Extract the leaderboard..."
     update_frequency: "weekly"
   ```

2. **Create Scraper Config** (`configs/scrapers/new_site.json`):
   ```json
   {
     "name": "new_site",
     "type": "html_file",
     "source": {
       "type": "local_file",
       "path": "data/benchmark_files/NewSite.html"
     }
   }
   ```

3. **Test Update**:
   ```bash
   python scripts/update_claude_benchmarks.py --sites new_site --force
   ```

## Troubleshooting

### Common Issues

1. **Claude CLI Not Found**:
   - Ensure Claude Code is installed: `pip install claude-cli`
   - Check PATH includes Claude installation

2. **Timeout Errors**:
   - Increase timeout in configuration
   - Some sites may take longer to render

3. **Invalid HTML Output**:
   - Check extraction prompt clarity
   - Verify site structure hasn't changed

4. **No Data Extracted**:
   - Review table selector in scraper config
   - Check column name mappings

### Debug Commands

```bash
# Test with dry run
python scripts/update_claude_benchmarks.py --dry-run --verbose

# Generate manual instructions
python scripts/update_claude_benchmarks.py --manual

# Force update single site
python scripts/update_claude_benchmarks.py --sites superclue --force --verbose

# Check scraper output
python scripts/run.py collect-all --scrapers superclue --verbose
```

## Future Improvements

- [ ] API integration instead of subprocess
- [ ] Automatic detection of browser interaction requirements
- [ ] Visual diff of HTML changes
- [ ] Metrics dashboard for update success rates
- [ ] Automatic fallback to alternative sources