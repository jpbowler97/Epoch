# Task: Update Complex Benchmark Data

You need to fetch and save benchmark data from JavaScript-rendered websites.
Use your WebFetch tool to get the rendered content, then save it as HTML files.

## Instructions:

### 1. Update superclue

1. Use WebFetch to get content from: https://www.superclueai.com/
2. Extract the following data:
   Extract the complete leaderboard table with:
   - Model names (模型名称)
   - Organization/Developer (机构)
   - Overall score (总分)
   - Math scores (数学)
   - Reasoning scores (推理)
   - Code generation scores (代码)
   - Agent capabilities (智能体)
   Format as a clean HTML table with proper <table>, <thead>, <tbody> structure.
   Ensure all Chinese and English text is preserved correctly.
   
3. Create a clean HTML file with:
   - DOCTYPE and proper HTML structure
   - A <table> element with the extracted data
   - Proper <thead> and <tbody> sections
   - All column headers and data preserved
4. Save to: data/benchmark_files/SuperCLUE.html
5. Add timestamp comment: <!-- Updated: 2025-08-15T16:21:36.781542 -->

### 2. Update physics_iq

1. Use WebFetch to get content from: https://physics-iq.github.io/
2. Extract the following data:
   Extract the physics understanding benchmark table with:
   - Model names (e.g., VideoPoet, Lumiere, Sora)
   - Physics IQ scores (percentage values)
   - Model types (i2v, multiframe, etc.)
   Format as HTML table with clear column headers.
   
3. Create a clean HTML file with:
   - DOCTYPE and proper HTML structure
   - A <table> element with the extracted data
   - Proper <thead> and <tbody> sections
   - All column headers and data preserved
4. Save to: data/benchmark_files/PhysicsIQ.html
5. Add timestamp comment: <!-- Updated: 2025-08-15T16:21:36.781565 -->

### 3. Update olympic_arena

1. Use WebFetch to get content from: https://gair-nlp.github.io/OlympicArena
2. Extract the following data:
   Extract the complete leaderboard table with:
   - Model names (e.g., GPT-4o, Claude-3.5-Sonnet)
   - Developer/Organization
   - Overall scores (percentage)
   - Subject-specific scores: Math, Physics, Chemistry, Biology, Geography, Astronomy, CS
   Include both validation and test set results if available.
   
3. Create a clean HTML file with:
   - DOCTYPE and proper HTML structure
   - A <table> element with the extracted data
   - Proper <thead> and <tbody> sections
   - All column headers and data preserved
4. Save to: data/benchmark_files/OlympicArena.html
5. Add timestamp comment: <!-- Updated: 2025-08-15T16:21:36.781572 -->

### 4. Update video_arena

1. Use WebFetch to get content from: https://artificialanalysis.ai/text-to-video/arena
2. Extract the following data:
   Extract the text-to-video model arena leaderboard with:
   - Model names (e.g., Veo, Sora, Runway Gen3)
   - ELO scores
   - Quality ratings
   - Win rates if available
   Format as HTML table with all ranking information.
   
3. Create a clean HTML file with:
   - DOCTYPE and proper HTML structure
   - A <table> element with the extracted data
   - Proper <thead> and <tbody> sections
   - All column headers and data preserved
4. Save to: data/benchmark_files/VideoArena.html
5. Add timestamp comment: <!-- Updated: 2025-08-15T16:21:36.781576 -->

## Output Requirements:
- Each HTML file must be valid HTML with proper structure
- Tables must have clear headers matching the data
- Preserve all numeric scores exactly as found
- Include model names without modification

## Important:
- If a site requires interaction or returns no data, log the issue
- Create the HTML files even if you only get partial data
- Ensure files are saved in the exact paths specified