"""Model name normalization utilities."""

import re
from typing import Dict, Set


# Common date patterns to remove (after underscores conversion)
DATE_PATTERNS = [
    r'_\d{4}_\d{2}_\d{2}',          # _2025_02_27
    r'_\d{8}',                      # _20250514
    r'_\d{4}\d{2}\d{2}',           # _20250514 (alternative)
    r'_\d{4}$',                    # _2025 (year only at end)
    r'_\d{4}_\d{2}$',              # _2025_02 (year-month at end)
    r'_\d{6}$',                    # _202502 (compact year-month)
    r'_0[1-9]\d{2}',               # _0709, _0528, _0301 (month-day patterns starting with 0)
    r'_1[0-2]\d{2}',               # _1201, _1025 (month-day patterns starting with 10-12)
]

# Common trailing patterns to remove (in order of specificity, after underscores conversion)
# NOTE: Be conservative - only remove clear version/date suffixes, not model variants
TRAILING_PATTERNS = [
    r'_\d+k_exp$',                 # _16k_exp
    r'_exp_\d{2}_\d{2}$',         # _exp_02_05
    r'_preview_\d{2}_\d{2}$',     # _preview_02_24
    r'_exp$',                      # _exp
    r'_chat$',                     # _chat
    r'_instruct$',                 # _instruct
    r'_base$',                     # _base
    r'_it$',                       # _it (instruction tuned)
    # Remove only numeric suffixes that look like versions, not model identifiers
    r'_\d+\.\d+$',                # _1.0, _2.5 (version numbers with decimals)
]

# Patterns for version removal that should NOT apply to core model identifiers
VERSION_REMOVAL_PATTERNS = [
    r'_v\d+\.?\d*$',              # _v1.0, _v2 - but NOT for core model families
]

# Model families where version numbers are part of the core identifier
CORE_MODEL_FAMILIES = {
    'deepseek', 'qwen', 'qwen2', 'qwen3', 'gpt', 'claude', 'gemini', 'grok', 'llama'
}

# Special model family mappings for complex cases
SPECIAL_MAPPINGS: Dict[str, str] = {
    # OpenAI models
    'chatgpt-4o-latest': 'gpt_4o',
    'gpt-4.5-preview': 'gpt_4.5_preview',
    'gpt-4.1': 'gpt_4.1',
    'gpt-4.1-2025-04-14': 'gpt_4.1',
    'gpt-4o-mini': 'gpt_4o_mini',
    'gpt-4-turbo': 'gpt_4_turbo',
    'gpt-3.5-turbo': 'gpt_3.5_turbo',
    
    # Anthropic models
    'claude-opus-4': 'claude_opus_4',
    'claude-sonnet-4': 'claude_sonnet_4',
    'claude-3.7-sonnet': 'claude_3.7_sonnet',
    'claude-3.5-sonnet': 'claude_3.5_sonnet',
    'claude-3.5-haiku': 'claude_3.5_haiku',
    'claude-3-opus': 'claude_3_opus',
    'claude-3-sonnet': 'claude_3_sonnet',
    'claude-3-haiku': 'claude_3_haiku',
    
    # Google models
    'gemini-2.5-pro': 'gemini_2.5_pro',
    'gemini-2.5-flash': 'gemini_2.5_flash',
    'gemini-2.0-pro': 'gemini_2.0_pro',
    'gemini-2.0-flash': 'gemini_2.0_flash',
    'gemini-1.5-pro': 'gemini_1.5_pro',
    'gemini-1.5-flash': 'gemini_1.5_flash',
    
    # xAI models
    'grok-4': 'grok_4',
    'grok-3': 'grok_3',
    'grok-2': 'grok_2',
    'grok-3-mini': 'grok_3_mini',
    'grok-2-mini': 'grok_2_mini',
    
    # DeepSeek models
    'deepseek-r1': 'deepseek_r1',
    'deepseek-v3': 'deepseek_v3',
    'deepseek-v2.5': 'deepseek_v2.5',
    'deepseek-coder-v2': 'deepseek_coder_v2',
    
    # Qwen models
    'qwen3-235b': 'qwen3_235b',
    'qwen3-480b': 'qwen3_480b',
    'qwen3-32b': 'qwen3_32b',
    'qwen2.5-max': 'qwen2.5_max',
    'qwen2.5-72b': 'qwen2.5_72b',
    'qwen2.5-coder-32b': 'qwen2.5_coder_32b',
    
    # Other models
    'kimi-k2': 'kimi_k2',
    'o1': 'o1',
    'o3': 'o3',
    'o4-mini': 'o4_mini',
    'o3-mini': 'o3_mini',
    'llama-3.1-405b': 'llama_3.1_405b',
    'llama-3.1-70b': 'llama_3.1_70b',
    'llama-3.1-8b': 'llama_3.1_8b',
    'llama-3.3-70b': 'llama_3.3_70b',
    'llama-4-maverick-17b': 'llama_4_maverick_17b',
    'llama-4-scout-17b': 'llama_4_scout_17b',
}


def normalize_model_name(name: str) -> str:
    """Normalize model name by standardizing format and removing dates/versions.
    
    Args:
        name: Original model name
        
    Returns:
        Normalized model name
        
    Examples:
        >>> normalize_model_name("grok-4-0709")
        'grok_4'
        >>> normalize_model_name("gpt-4.5-preview-2025-02-27")
        'gpt_4.5_preview'
        >>> normalize_model_name("claude-opus-4-20250514-thinking-16k")
        'claude_opus_4_thinking_16k'
        >>> normalize_model_name("kimi-k2-0711-preview")
        'kimi_k2_preview'
    """
    if not name:
        return name
    
    # Check for special mappings first (exact match before any normalization)
    original_lower = name.lower().strip()
    if original_lower in SPECIAL_MAPPINGS:
        return SPECIAL_MAPPINGS[original_lower]
    
    # Convert to lowercase and remove parentheses and their contents
    normalized = original_lower
    normalized = re.sub(r'\([^)]*\)', '', normalized)
    
    # Replace dashes and spaces with underscores
    normalized = re.sub(r'[-\s]+', '_', normalized)
    
    # Handle version patterns BEFORE date removal to preserve model identifiers
    # Look for patterns like "deepseek_v3_0324" and preserve the version part
    version_preservation_patterns = [
        (r'(deepseek)_v(\d+(?:\.\d+)?)_\d{4}', r'\1_v\2'),  # deepseek_v3_0324, deepseek_v2.5_1210 -> deepseek_v3, deepseek_v2.5
        (r'(qwen\d*)_v(\d+(?:\.\d+)?)_\d+', r'\1_v\2'),   # qwen3_v2_0324 -> qwen3_v2
        (r'(gpt)_(\d+(?:\.\d+)?)_\d{4}_\d{2}_\d{2}', r'\1_\2'),  # gpt_4.1_2025_04_14 -> gpt_4.1
    ]
    
    for pattern, replacement in version_preservation_patterns:
        normalized = re.sub(pattern, replacement, normalized)
    
    # Remove date patterns after version preservation
    for pattern in DATE_PATTERNS:
        normalized = re.sub(pattern, '', normalized)
    
    # Remove trailing patterns (in order of specificity)
    for pattern in TRAILING_PATTERNS:
        normalized = re.sub(pattern, '', normalized)
    
    # Handle version removal patterns - only apply if NOT a core model family
    base_name = normalized.split('_')[0]
    if base_name not in CORE_MODEL_FAMILIES:
        for pattern in VERSION_REMOVAL_PATTERNS:
            normalized = re.sub(pattern, '', normalized)
    
    # Clean up multiple consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    # Handle special model-specific patterns
    # Remove common suffixes that indicate variants
    variant_patterns = [
        r'_a\d+b$',           # _a22b, _a35b
        r'_no_thinking$',     # _no_thinking
        r'_thinking$',        # _thinking  
        r'_high$',            # _high
        r'_mini_high$',       # _mini_high
        r'_beta$',            # _beta
        r'_latest$',          # _latest
    ]
    
    for pattern in variant_patterns:
        normalized = re.sub(pattern, '', normalized)
    
    # Clean up again after removals
    normalized = re.sub(r'_+', '_', normalized)
    normalized = normalized.strip('_')
    
    # Final cleanup: ensure only valid characters
    normalized = re.sub(r'[^a-z0-9._]', '_', normalized)
    normalized = re.sub(r'_+', '_', normalized)
    normalized = normalized.strip('_')
    
    return normalized


def get_model_family(normalized_name: str) -> str:
    """Extract model family from normalized name.
    
    Args:
        normalized_name: Normalized model name
        
    Returns:
        Model family identifier
        
    Examples:
        >>> get_model_family("gpt_4.5_preview")
        'gpt_4'
        >>> get_model_family("claude_opus_4_thinking_16k")
        'claude_opus_4'
        >>> get_model_family("llama_3.1_405b")
        'llama_3.1'
    """
    # Split by underscores and take meaningful parts
    parts = normalized_name.split('_')
    
    # Common family patterns
    if len(parts) >= 2:
        if parts[0] in ['gpt', 'claude', 'gemini', 'grok', 'llama', 'qwen', 'deepseek']:
            if len(parts) >= 3 and parts[2] in ['opus', 'sonnet', 'haiku', 'pro', 'flash', 'mini']:
                return '_'.join(parts[:3])
            return '_'.join(parts[:2])
    
    return parts[0] if parts else normalized_name


def detect_potential_duplicates(names: Set[str]) -> Dict[str, Set[str]]:
    """Detect potential duplicate models with different naming.
    
    Args:
        names: Set of model names
        
    Returns:
        Dictionary mapping normalized names to sets of original names
    """
    duplicates = {}
    
    for name in names:
        normalized = normalize_model_name(name)
        if normalized not in duplicates:
            duplicates[normalized] = set()
        duplicates[normalized].add(name)
    
    # Filter to only return actual duplicates
    return {k: v for k, v in duplicates.items() if len(v) > 1}