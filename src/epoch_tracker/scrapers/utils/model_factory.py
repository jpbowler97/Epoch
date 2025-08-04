"""Model creation utilities for consistent model object generation."""

import logging
from typing import Dict, Optional, Any, List
from datetime import datetime

from ...models import Model


class ModelFactory:
    """Factory for creating standardized Model objects from scraped data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Developer detection patterns
        self.developer_patterns = {
            # Major AI companies
            'openai': ['gpt', 'chatgpt', 'o1', 'o3', 'davinci', 'curie', 'babbage', 'sora'],
            'anthropic': ['claude'],
            'google': ['gemini', 'palm', 'bard', 'lamda', 'chinchilla', 'veo'],
            'meta': ['llama', 'opt'],
            'microsoft': ['phi', 'turing'],
            'mistral': ['mistral', 'mixtral', 'pixtral', 'magistral'],
            'deepseek': ['deepseek'],
            'alibaba': ['qwen', 'tongyi'],
            '01 ai': ['yi'],
            'bytedance': ['doubao'],
            'moonshot': ['kimi'],
            'xai': ['grok'],
            'cohere': ['command'],
            'ai21 labs': ['jamba'],
            'reka ai': ['reka'],
            'nexusflow': ['athene'],
            'databricks': ['dbrx'],
            'snowflake': ['arctic'],
            'zhipu ai': ['glm', 'chatglm'],
            'tencent': ['hunyuan'],
            'minimax': ['minimax'],
            'stepfun': ['step'],
            'amazon': ['nova'],
            'nvidia': ['nemotron'],
            'openchat': ['openchat'],
            'huawei': ['pangu'],
            # Video generation companies
            'runway': ['runway', 'gen_3', 'gen3'],
            'stability ai': ['stable'],
            'pika labs': ['pika']
        }
    
    def create_model(self, 
                    name: str,
                    developer: Optional[str] = None,
                    benchmarks: Optional[Dict[str, float]] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    sources: Optional[List[str]] = None,
                    source_url: Optional[str] = None,
                    source_description: Optional[str] = None,
                    **kwargs) -> Optional[Model]:
        """Create a Model object with standardized processing.
        
        Args:
            name: Model name
            developer: Developer organization (auto-detected if None)
            benchmarks: Dictionary of benchmark_name -> score
            metadata: Additional metadata
            sources: List of source URLs/descriptions
            source_url: Primary source URL
            source_description: Description of the source
            **kwargs: Additional Model fields
            
        Returns:
            Model object or None if creation fails
        """
        if not name or not name.strip():
            self.logger.warning("Cannot create model with empty name")
            return None
        
        name = name.strip()
        
        # Auto-detect developer if not provided
        if not developer:
            developer = self.detect_developer(name)
        
        # Clean up developer name
        developer = developer or 'Unknown'
        
        # Initialize metadata
        model_metadata = metadata or {}
        
        # Create base model
        try:
            model = Model(
                name=name,
                developer=developer,
                metadata=model_metadata,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Failed to create model '{name}': {e}")
            return None
        
        # Add benchmarks
        if benchmarks:
            for benchmark_name, score in benchmarks.items():
                if score is not None:
                    model.add_benchmark(benchmark_name, score)
        
        # Add sources
        if sources:
            for source in sources:
                if source:
                    model.sources.append(source)
        
        # Add primary source
        if source_url or source_description:
            model.add_source(source_url or '', source_description or '')
        
        return model
    
    def detect_developer(self, model_name: str) -> Optional[str]:
        """Auto-detect developer from model name using pattern matching.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Detected developer name or None
        """
        name_lower = model_name.lower()
        
        # Check each developer's patterns
        for developer, patterns in self.developer_patterns.items():
            for pattern in patterns:
                if pattern in name_lower:
                    # Return properly capitalized developer name
                    return self._format_developer_name(developer)
        
        self.logger.debug(f"Could not auto-detect developer for model: {model_name}")
        return None
    
    def _format_developer_name(self, developer_key: str) -> str:
        """Format developer name for consistency.
        
        Args:
            developer_key: Internal developer key
            
        Returns:
            Properly formatted developer name
        """
        # Special cases for proper formatting
        formatting_map = {
            'openai': 'OpenAI',
            'anthropic': 'Anthropic',
            'google': 'Google',
            'meta': 'Meta',
            'microsoft': 'Microsoft',
            'mistral': 'Mistral',
            'deepseek': 'DeepSeek',
            'alibaba': 'Alibaba',
            '01 ai': '01 AI',
            'bytedance': 'ByteDance',
            'moonshot': 'Moonshot',
            'xai': 'xAI',
            'cohere': 'Cohere',
            'ai21 labs': 'AI21 Labs',
            'reka ai': 'Reka AI',
            'nexusflow': 'NexusFlow',
            'databricks': 'Databricks',
            'snowflake': 'Snowflake',
            'zhipu ai': 'Zhipu AI',
            'tencent': 'Tencent',
            'minimax': 'MiniMax',
            'stepfun': 'StepFun',
            'amazon': 'Amazon',
            'nvidia': 'Nvidia',
            'openchat': 'OpenChat',
            'huawei': 'Huawei',
            # Video generation companies
            'runway': 'Runway',
            'stability ai': 'Stability AI',
            'pika labs': 'Pika Labs'
        }
        
        return formatting_map.get(developer_key, developer_key.title())
    
    def create_benchmarks_dict(self, 
                              raw_scores: Dict[str, Any],
                              benchmark_mapping: Dict[str, str],
                              source_prefix: str = '') -> Dict[str, float]:
        """Create standardized benchmarks dictionary from raw scores.
        
        Args:
            raw_scores: Dictionary of raw score data
            benchmark_mapping: Mapping of benchmark_name -> raw_field_name
            source_prefix: Prefix to add to benchmark names
            
        Returns:
            Dictionary of benchmark_name -> numeric_score
        """
        from .numeric_parser import NumericParser
        parser = NumericParser()
        
        benchmarks = {}
        
        for benchmark_name, raw_field in benchmark_mapping.items():
            if raw_field in raw_scores:
                score = parser.parse_numeric(raw_scores[raw_field])
                if score is not None:
                    # Add source prefix if provided
                    full_name = f"{source_prefix}_{benchmark_name}" if source_prefix else benchmark_name
                    benchmarks[full_name] = score
        
        return benchmarks
    
    def create_metadata_dict(self, 
                           raw_data: Dict[str, Any],
                           metadata_mapping: Dict[str, str],
                           source_name: str = '') -> Dict[str, Any]:
        """Create metadata dictionary from raw data.
        
        Args:
            raw_data: Dictionary of raw data
            metadata_mapping: Mapping of metadata_field -> raw_field_name
            source_name: Name of the data source
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Add source tracking
        if source_name:
            metadata['source'] = source_name
            metadata['scraped_at'] = datetime.utcnow().isoformat()
        
        # Map fields from raw data
        for metadata_field, raw_field in metadata_mapping.items():
            if raw_field in raw_data and raw_data[raw_field] is not None:
                value = raw_data[raw_field]
                # Clean string values
                if isinstance(value, str):
                    value = value.strip()
                    if value:  # Only add non-empty strings
                        metadata[metadata_field] = value
                else:
                    metadata[metadata_field] = value
        
        return metadata
    
    def batch_create_models(self, 
                           rows_data: List[Dict[str, Any]],
                           column_config: Dict[str, Any],
                           source_config: Dict[str, Any]) -> List[Model]:
        """Create multiple models from a batch of row data.
        
        Args:
            rows_data: List of dictionaries containing row data
            column_config: Configuration for column mapping and processing
            source_config: Configuration for source information
            
        Returns:
            List of created Model objects
        """
        models = []
        source_name = source_config.get('name', 'unknown')
        
        for i, row_data in enumerate(rows_data):
            try:
                model = self._create_model_from_row(row_data, column_config, source_config)
                if model:
                    models.append(model)
            except Exception as e:
                self.logger.warning(f"Failed to create model from row {i} in {source_name}: {e}")
                continue
        
        self.logger.info(f"Created {len(models)} models from {len(rows_data)} rows for source {source_name}")
        return models
    
    def _create_model_from_row(self, 
                              row_data: Dict[str, Any],
                              column_config: Dict[str, Any],
                              source_config: Dict[str, Any]) -> Optional[Model]:
        """Create a single model from row data and configuration."""
        from .column_mapper import ColumnMapper
        
        mapper = ColumnMapper()
        
        # Extract basic fields
        columns = column_config.get('columns', {})
        model_name = mapper.extract_value(row_data, columns.get('model'))
        developer = mapper.extract_value(row_data, columns.get('developer'))
        
        if not model_name:
            return None
        
        # Extract benchmarks
        benchmark_mapping = column_config.get('benchmarks', {})
        benchmarks = self.create_benchmarks_dict(
            row_data, 
            benchmark_mapping, 
            source_config.get('name', '')
        )
        
        # Extract metadata
        metadata_mapping = column_config.get('metadata', {})
        metadata = self.create_metadata_dict(
            row_data,
            metadata_mapping,
            source_config.get('name', '')
        )
        
        # Create model
        return self.create_model(
            name=model_name,
            developer=developer,
            benchmarks=benchmarks,
            metadata=metadata,
            source_url=source_config.get('source_url'),
            source_description=source_config.get('description')
        )