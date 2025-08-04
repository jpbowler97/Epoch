# Adding Benchmark References - Complete Workflow Guide

This guide provides a step-by-step process for adding new benchmark references to the FLOP estimation system. The entire process is designed to be seamless through a single JSON configuration file.

## Quick Start - Adding a New Benchmark

**TL;DR**: Edit `configs/benchmark_references.json` and add your benchmark data - everything else works automatically.

### 1. **Add Reference Data to JSON Config**

Edit `configs/benchmark_references.json` and add a new entry to the `benchmark_references` section:

```json
{
  "benchmark_references": {
    "your_new_benchmark": {
      "reference_model": "llama_3.1_405b",
      "reference_score": 85.4,
      "reference_flop": 3.8e25,
      "source": "Your Benchmark Leaderboard + Epoch AI disclosure analysis",
      "source_url": "https://your-benchmark.com/leaderboard",
      "scaling_exponent": 2.5,
      "benchmark_type": "percentage",
      "notes": "Description of what this benchmark measures"
    }
  }
}
```

### 2. **Test the Integration**

```bash
# Test that the new benchmark loads correctly
python -c "
from scripts.data_processing.estimate_flops import load_benchmark_references
refs = load_benchmark_references()
print('Available benchmarks:', list(refs.keys()))
print('Your new benchmark:', refs.get('your_new_benchmark'))
"

# Run FLOP estimation to see it in action
python scripts/data_processing/estimate_flops.py --dry-run --verbose | grep "your_new_benchmark"
```

### 3. **Verify Results**

The system will automatically:
- ✅ Load your new benchmark configuration
- ✅ Use it for multi-benchmark FLOP estimation  
- ✅ Include it in confidence calculations
- ✅ Display it in estimation reasoning

**That's it!** No code changes required.

---

## Detailed Configuration Guide

### Field Descriptions

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `reference_model` | ✅ | string | Name of the reference model (usually "llama_3.1_405b") |
| `reference_score` | ✅ | number | Actual benchmark score for the reference model |
| `reference_flop` | ✅ | number | Training FLOP of the reference model (scientific notation: 3.8e25) |
| `source` | ✅ | string | Citation for benchmark score + FLOP estimate |
| `source_url` | ⚪ | string | URL to the benchmark leaderboard/paper |
| `scaling_exponent` | ✅ | number | Power law scaling factor (2.0-3.0) |
| `benchmark_type` | ⚪ | string | Type: "elo_rating", "percentage", etc. |
| `notes` | ⚪ | string | Description of what the benchmark measures |

### Scaling Exponent Guidelines

Choose based on the correlation between benchmark performance and compute:

```json
{
  "scaling_guidelines": {
    "3.0": {
      "description": "Strong correlation with compute",
      "use_cases": ["General performance ELO scores", "Overall capability ratings"],
      "examples": ["lmarena_score", "openlm_arena_elo"]
    },
    "2.5": {
      "description": "Moderate correlation with compute", 
      "use_cases": ["Reasoning tasks", "Complex benchmarks"],
      "examples": ["mmlu_pro_score", "math_benchmarks"]
    },
    "2.0": {
      "description": "Weaker correlation with compute",
      "use_cases": ["Safety scores", "Specialized metrics"],
      "examples": ["aai_score", "helpfulness_ratings"]
    }
  }
}
```

### Example Configurations

#### **Vision Benchmark**
```json
"vision_score": {
  "reference_model": "gpt_4o",
  "reference_score": 88.5,
  "reference_flop": 3.8e25,
  "source": "Vision Benchmark Dataset + Epoch AI FLOP estimates", 
  "source_url": "https://vision-benchmark.ai/",
  "scaling_exponent": 2.5,
  "benchmark_type": "percentage",
  "notes": "Multimodal vision and reasoning capabilities"
}
```

#### **Math Reasoning Benchmark**
```json
"math_reasoning_score": {
  "reference_model": "llama_3.1_405b",
  "reference_score": 76.8,
  "reference_flop": 3.8e25,
  "source": "Math Reasoning Benchmark + Epoch AI disclosure analysis",
  "source_url": "https://math-benchmark.org/leaderboard", 
  "scaling_exponent": 2.5,
  "benchmark_type": "percentage",
  "notes": "Advanced mathematical reasoning and problem solving"
}
```

#### **Code Generation Benchmark**
```json
"code_generation_score": {
  "reference_model": "llama_3.1_405b", 
  "reference_score": 82.1,
  "reference_flop": 3.8e25,
  "source": "HumanEval+ Benchmark + Epoch AI disclosure analysis",
  "source_url": "https://github.com/evalplus/evalplus",
  "scaling_exponent": 3.0,
  "benchmark_type": "percentage", 
  "notes": "Code generation and programming capabilities"
}
```

---

## Finding Reference Data

### 1. **Get Reference Model Score**

You need the benchmark score for your chosen reference model. 

**For Llama 3.1 405B:**
1. Check scraped data: `data/scraped/openlm_arena_models.json` or `data/scraped/lmarena_models.json`
2. Search benchmark leaderboards manually
3. Check official papers/announcements

**Example search in scraped data:**
```bash
# Search for Llama 405B benchmark scores
grep -A 20 "llama_3.1_405b" data/scraped/openlm_arena_models.json | grep "your_benchmark_name"
```

### 2. **Choose Appropriate Reference Model**

| Use Case | Recommended Reference | Why |
|----------|----------------------|-----|
| **General benchmarks** | `llama_3.1_405b` | Well-characterized, high-quality FLOP estimate |
| **Vision benchmarks** | `gpt_4o` | Strong vision capabilities |
| **Long context** | `claude_3_opus` | Excellent long context handling |
| **Coding tasks** | `llama_3.1_405b` | Good coding performance, reliable FLOP |

### 3. **Verify FLOP Estimates**

Current high-confidence FLOP estimates:
- **Llama 3.1 405B**: 3.8e25 (Epoch AI + Meta disclosure)
- **GPT-4o**: 3.8e25 (Epoch AI estimate)  
- **Claude 3 Opus**: 1.6e25 (Epoch AI estimate)
- **Grok-2**: 3.0e25 (Epoch AI + xAI disclosure)

---

## Integration Workflow

### Development Process

1. **Collect Data**: Get benchmark score for reference model
2. **Edit Config**: Add entry to `configs/benchmark_references.json`
3. **Test Locally**: Run estimation dry-run to verify 
4. **Commit Changes**: The JSON file is version-controlled
5. **Deploy**: Changes take effect immediately

### Validation Steps

```bash
# 1. Validate JSON syntax
python -c "import json; json.load(open('configs/benchmark_references.json'))"

# 2. Test benchmark loading
python -c "
from scripts.data_processing.estimate_flops import load_benchmark_references
refs = load_benchmark_references()
print('New benchmark loaded:', 'your_benchmark_name' in refs)
"

# 3. Run estimation test
python scripts/data_processing/estimate_flops.py --dry-run | head -20

# 4. Check specific model with new benchmark
python -c "
from scripts.data_processing.estimate_flops import estimate_from_multiple_benchmarks
# Test with model that has your benchmark score
"
```

### Production Deployment

1. **Edit JSON**: Make changes to `configs/benchmark_references.json`
2. **Commit**: Changes are automatically versioned
3. **Deploy**: System loads new config automatically
4. **Verify**: Check logs for successful loading

---

## Troubleshooting

### Common Issues

**❌ JSON Syntax Error**
```bash
# Error: json.decoder.JSONDecodeError
# Fix: Validate JSON syntax
python -c "import json; json.load(open('configs/benchmark_references.json'))"
```

**❌ Missing Reference Score**
```bash
# Error: Benchmark score not found for reference model
# Fix: Add score to JSON or verify model name in scraped data
grep -r "your_reference_model" data/scraped/
```

**❌ Poor Estimation Quality**
```bash
# Error: Estimates seem off
# Fix: Adjust scaling_exponent (try values between 2.0-3.0)
# Check reference score accuracy
```

### Debug Commands

```bash
# Check current benchmark configuration
python -c "
from scripts.data_processing.estimate_flops import BENCHMARK_REFERENCE_MODELS
for name, config in BENCHMARK_REFERENCE_MODELS.items():
    print(f'{name}: {config[\"reference_score\"]} (α={config[\"scaling_exponent\"]})')
"

# Test single benchmark estimation
python -c "
from scripts.data_processing.estimate_flops import estimate_from_single_benchmark
from epoch_tracker.estimation import ComputeEstimator
estimator = ComputeEstimator()
result = estimate_from_single_benchmark('your_benchmark', 85.0, estimator)
print('Estimate:', result)
"

# Check for models with your benchmark
grep -r "your_benchmark_name" data/scraped/ | head -5
```

---

## Maintaining Reference Data

### When to Update

- ✅ **New benchmark types** become available
- ✅ **Reference models get updated scores** on leaderboards  
- ✅ **Better FLOP estimates** become available for reference models
- ✅ **Scaling relationships** are empirically validated

### Best Practices

1. **Document sources**: Always include URLs and citations
2. **Version control**: All changes go through Git
3. **Test thoroughly**: Validate with known models
4. **Monitor quality**: Check estimation accuracy over time
5. **Update periodically**: Refresh reference scores quarterly

### Configuration Schema Validation

The JSON configuration follows this schema:

```json
{
  "type": "object",
  "required": ["benchmark_references"],
  "properties": {
    "benchmark_references": {
      "type": "object",
      "patternProperties": {
        "^[a-z_]+$": {
          "type": "object",
          "required": ["reference_model", "reference_score", "reference_flop", "source", "scaling_exponent"],
          "properties": {
            "reference_model": {"type": "string"},
            "reference_score": {"type": "number"},
            "reference_flop": {"type": "number"},
            "source": {"type": "string"},
            "scaling_exponent": {"type": "number", "minimum": 1.0, "maximum": 5.0}
          }
        }
      }
    }
  }
}
```

---

## Summary

The benchmark reference system is designed for **zero-friction addition** of new benchmarks:

1. ✅ **Single file edit**: Only touch `configs/benchmark_references.json`
2. ✅ **Automatic integration**: System loads new config instantly  
3. ✅ **Robust fallbacks**: Never breaks if config has issues
4. ✅ **Full documentation**: Every field is explained
5. ✅ **Version controlled**: All changes are tracked
6. ✅ **Validation ready**: Built-in testing and debugging

**Result**: Adding new benchmark types takes minutes, not hours, and requires no code changes.