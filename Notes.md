# Notes

## Epoch's Current >1e25 FLOP Tracker Analysis

Source: https://epoch.ai/data-insights/models-over-1e25-flop

### Data Schema (Current)
- **Model name**
- **Training FLOP estimate** 
- **Confidence level** (speculative vs high-precision)
- **Inclusion justification** (reasoning for inclusion/exclusion)

### FLOP Estimation Methods
1. **Direct Calculation**:
   - Parameter count � Dataset size � Training details
   - Hardware specifications and training duration
   
2. **Benchmark-based Imputation**:
   - Sigmoid curves relating compute to benchmark performance
   - Median estimate across multiple benchmarks
   - Cross-validation accuracy within 2.2x of true compute
   - Noted as "fairly speculative"

### Data Sources
- **AI Labs**: Google, Meta, Microsoft, OpenAI, Anthropic, etc.
- **Model Repositories**: HuggingFace, etc.
- **Benchmark Leaderboards**: 
  - Chatbot Arena
  - HELM  
  - CompassRank, CompassArena (China-focused)

### Key Methodology Rules
- Exclude fine-tuned variants (focus on base models)
- Use qualitative performance assessment for image/video models
- Focus on publicly disclosed models
- Maintain ongoing tracking and updates

### Presentation Format
- Tabular with model/compute/confidence columns
- Detailed methodology explanations
- Separate confirmed vs potential model lists

## FLOP Calculation References

### Basic Formula
Training FLOP H Parameters � Tokens � 2

### Considerations for Our Schema
- Need both training and inference compute tracking
- Confidence levels important for uncertain estimates
- Multiple estimation methods when data allows
- Clear reasoning/justification fields
- Source attribution and update timestamps

## Starting Examples for Implementation

### Claude 3.5 Sonnet (Anthropic)
- **Parameters**: >175 billion (estimated)
- **Training FLOP**: Not publicly disclosed by Anthropic
- **Benchmarks Available**: 
  - GPQA (graduate-level reasoning)
  - MMLU (undergraduate knowledge) 
  - HumanEval (coding proficiency)
  - SWE-bench Verified: 49.0%
  - Multilingual Math: 91.6%
  - Reasoning Over Text: 87.1%
- **Release Date**: 2024
- **Estimation Method**: Will require benchmark-based imputation
- **Context**: 200K tokens, $3/$15 per million tokens
- **Performance**: 2x speed of Claude 3 Opus
- **Sources**:
  - https://www.anthropic.com/news/claude-3-5-sonnet (official announcement)
  - https://medium.com/accredian/claude-3-5-sonnet-setting-new-ai-benchmarks-a72c13ad3d3e (benchmark compilation)
  - https://www.vellum.ai/blog/claude-3-5-sonnet-vs-gpt4o (performance comparison)

### Llama 3.1 405B (Meta)  
- **Parameters**: 405 billion (confirmed)
- **Training FLOP**: 3.8 × 10^25 FLOPs (publicly disclosed)
- **Training Infrastructure**: 16,000+ H100 GPUs  
- **Training Data**: 15+ trillion tokens
- **Release Date**: July 2024
- **Estimation Method**: Direct calculation (complete data available)
- **Architecture**: Standard decoder-only transformer
- **Quantization**: BF16 to FP8 for inference
- **Status**: Largest open source model, world's largest openly available foundation model
- **Sources**:
  - https://ai.meta.com/blog/meta-llama-3-1/ (official Meta announcement)
  - https://huggingface.co/meta-llama/Llama-3.1-405B (model repository)
  - https://www.infoq.com/news/2024/07/meta-releases-llama31-405b/ (technical details)
  - https://www.interconnects.ai/p/llama-405b-open-frontier-model (analysis)

### GPT-4 (OpenAI)
- **Parameters**: 300-500B (estimated, not officially disclosed)
- **Training FLOP**: ~5.63e24 FLOP (estimated), likely >1e25 FLOP total
- **Training Cost**: $78 million worth of compute
- **Benchmarks Available**:
  - Simulated bar exam: top 10% of test takers
  - SWE-bench Verified: 33.2% (GPT-4o), 54.6% (GPT-4.1)
  - Scale's MultiChallenge: 38.3% (GPT-4.1)
- **Release Date**: March 2023
- **Architecture**: Mixture of Experts (MoE), sparse model >1 trillion parameters
- **Inference**: 128 GPUs, 8-way tensor parallelism, 16-way pipeline parallelism
- **Estimation Method**: Multiple estimates available, benchmark-based imputation possible
- **Status**: First model to exceed 1e25 FLOP threshold
- **Sources**:
  - https://openai.com/index/gpt-4-research/ (official OpenAI page)
  - https://epoch.ai/data-insights/models-over-1e25-flop (Epoch analysis)
  - https://semianalysis.com/2023/07/10/gpt-4-architecture-infrastructure/ (architecture analysis)
  - https://arxiv.org/abs/2303.08774 (GPT-4 Technical Report)

### Gemini Ultra (Google)
- **Parameters**: Not publicly disclosed
- **Training FLOP**: Not publicly disclosed
- **Training Infrastructure**: Large fleet of TPUv4 accelerators across multiple data centers
- **Benchmarks Available**:
  - MMLU: 90.0% (first model to outperform human experts)
  - MMMU: 59.4% (state-of-the-art multimodal benchmark)
  - Outperforms on 30 of 32 widely-used academic benchmarks
- **Release Date**: 2024
- **Architecture**: Trained on TPUv4/v5e infrastructure
- **Training Scale**: Expected to exceed GPT-4 pre-training FLOPs by 5x
- **Estimation Method**: Will require benchmark-based imputation
- **Sources**:
  - https://blog.google/technology/ai/google-gemini-ai/ (official Google announcement)
  - https://deepmind.google/models/gemini/ (Google DeepMind page)
  - https://www.semianalysis.com/p/google-gemini-eats-the-world-gemini (analysis)
  - https://www.datacenterdynamics.com/en/news/training-gemini-tpus-multiple-data-centers-and-risks-of-cosmic-rays/ (infrastructure details)

### Claude 3 Opus (Anthropic)
- **Parameters**: Not publicly disclosed
- **Training FLOP**: Not publicly disclosed
- **Context Window**: 200,000 tokens (expandable to 1 million)
- **Benchmarks Available**:
  - MMLU, GPQA, GSM8K: Leading performance across benchmarks
  - Needle In A Haystack: >99% accuracy
  - Generally outperforms GPT-4 on complex reasoning and coding tasks
- **Training Data**: Up to August 2023
- **Release Date**: March 2024
- **Estimation Method**: Will require benchmark-based imputation
- **Current Status**: Superseded by Claude 3.5 Sonnet and Claude 4 Opus
- **Sources**:
  - https://www.anthropic.com/news/claude-3-family (official Anthropic announcement)
  - https://www.anthropic.com/claude/opus (product page)
  - https://the-decoder.com/anthropic-unveils-new-claude-3-ai-models-to-beat-openai-and-google/ (launch analysis)
  - https://semianalysis.com/2024/12/11/scaling-laws-o1-pro-architecture-reasoning-training-infrastructure-orion-and-claude-3-5-opus-failures/ (technical analysis)

## Potential Data Sources for Scraping Pipeline

Based on manual research, here are consistent sources that could be good targets for automated scraping:

### 1. Hugging Face Open LLM Leaderboard
- **URL**: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- **Data Available**: Model rankings, parameter counts, benchmark scores
- **Advantages**:
  - Large collection of models with parameter counts
  - Standardized benchmark scores
  - Daily updates
  - API access via Hugging Face
- **Limitations**: Lacks training compute data, focuses mainly on open models
- **Scraping Feasibility**: High - API available

### 2. Chatbot Arena Leaderboard  
- **URL**: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard
- **Data Available**: ELO ratings, model versions, user preference data
- **Advantages**:
  - Real-world performance data
  - Regular updates
  - Covers both open and closed models
- **Limitations**: No parameter counts or training compute data
- **Scraping Feasibility**: Medium - structured but limited technical specs

### 3. LLM-Perf Leaderboard (Hugging Face)
- **URL**: https://huggingface.co/spaces/optimum/llm-perf-leaderboard  
- **Data Available**: Latency, throughput, memory usage, FLOP efficiency
- **Advantages**: 
  - Performance metrics including FLOP efficiency
  - Hardware-specific benchmarks
- **Limitations**: Focus on inference performance, not training compute
- **Scraping Feasibility**: Medium

### 4. Artificial Analysis Leaderboards
- **URL**: https://artificialanalysis.ai/leaderboards/models
- **Data Available**: Performance comparisons, pricing, speed metrics
- **Advantages**: Comprehensive model comparisons
- **Limitations**: Limited training compute information
- **Scraping Feasibility**: Medium

## Recommendation for Initial MVP Pipeline (Revised)

**Primary Sources:**

### 1. Hugging Face Open LLM Leaderboard ⭐ (PRIMARY RECOMMENDATION)
- **Why Start Here**: 
  - Daily updated model data with parameter counts
  - API access for reliable scraping
  - Covers wide range of models with benchmark scores
  - Can identify candidates for >1e25 FLOP estimation
- **Data Strategy**: Use parameter counts + benchmark scores for FLOP estimation using scaling laws

### 2. Chatbot Arena Leaderboard (SECONDARY)
- **Why Include**: 
  - Covers both open and closed models (GPT-4, Claude, Gemini)
  - Real performance data for benchmark-based estimation
  - Regular updates with model versions
- **Data Strategy**: Extract model names/versions, use for cross-referencing with other sources

### 3. Official Company Announcements (MANUAL + AUTOMATION)
- **Target Sources**:
  - Anthropic blog (Claude models)
  - Meta AI blog (Llama models) 
  - OpenAI research pages (GPT models)
  - Google DeepMind (Gemini models)
- **Data Strategy**: Scrape announcement pages for parameter counts, training details, benchmark results

**Validation Strategy**:
- Use Epoch AI tracker data to validate our scraped results
- Compare our FLOP estimates against Epoch's existing classifications
- Identify gaps where manual research is still needed

**Implementation Plan**:
1. Build Hugging Face API scraper first (reliable, structured)
2. Add Chatbot Arena scraper for closed model coverage  
3. Build company blog scrapers for official announcements
4. Validate all results against Epoch's existing data
5. Implement FLOP estimation logic based on collected data
4. Use Chatbot Arena for performance benchmarks and model version tracking


## Future Validation Ideas

### Cross-Validation Against Epoch Tracker
**Goal**: Systematically compare our automated findings against Epoch's manual process

**Implementation Approach**:
```bash
# Potential validation script  
python scripts/validate_against_epoch.py
```

**What it would do**:
- Scrape Epoch's current tracker at https://epoch.ai/data-insights/models-over-1e25-flop
- Compare our 9 identified models vs their official list
- Identify discrepancies: models we're missing, incorrect classifications
- Generate accuracy report with confidence intervals
- Analyze where automated vs manual approaches differ

**Key Validation Questions**:
- Do we capture all models Epoch has manually identified?
- Are our FLOP estimates within acceptable ranges of their assessments?
- What models are we finding that they haven't tracked yet?
- How do our confidence levels align with their classification certainty?

**Expected Outcomes**:
- Validation that our approach captures 90%+ of manually identified models
- Identification of edge cases requiring manual intervention
- Confidence in recommending our system to replace manual process
- Clear gaps analysis for future scraper development

**Timeline**: Medium priority - after expanding model coverage with Chatbot Arena and company scrapers

---

**Feedback based on user review**:
* Llama-3.1-405B is represented multiple times in the hugging face data as different models but really it's just one
  * Maybe we track variants differently?
* Can we remove .env.example file? and any dependencies on this?



## Questions for David
* I will be meeting with David Owens from Epoch to review progress later this week, if you think of helpful questions you think I should ask him to help clarify / improve the approach please add them here.

