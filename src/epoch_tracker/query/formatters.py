"""Output formatters for model query results."""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..models import Model


class BaseFormatter:
    """Base class for output formatters."""
    
    def format_models(self, models: List[Model], **kwargs) -> str:
        """Format a list of models for output."""
        raise NotImplementedError
    
    def _format_flop(self, flop: Optional[float]) -> str:
        """Format FLOP value for display."""
        if flop is None:
            return "Unknown"
        return f"{flop:.2e}"
    
    def _format_params(self, params: Optional[int]) -> str:
        """Format parameter count for display."""
        if params is None:
            return "Unknown"
        
        if params >= 1_000_000_000:
            return f"{params / 1_000_000_000:.0f}B"
        elif params >= 1_000_000:
            return f"{params / 1_000_000:.0f}M"
        else:
            return f"{params:,}"
    
    def _format_date(self, date: Optional[datetime]) -> str:
        """Format date for display."""
        if date is None:
            return "Unknown"
        return date.strftime("%Y-%m-%d")


class CompactFormatter(BaseFormatter):
    """Compact formatter for displaying model summaries."""
    
    def format_models(
        self, 
        models: List[Model], 
        show_reasoning: bool = False,
        title: str = "Models"
    ) -> str:
        """Format models in compact view."""
        if not models:
            return "No models found matching the criteria."
        
        lines = [f"=== {title} ===\n"]
        
        for i, model in enumerate(models, 1):
            lines.append(f"{i}. {model.name} ({model.developer})")
            
            # Core metrics
            flop_str = self._format_flop(model.training_flop)
            confidence = model.training_flop_confidence.value
            params_str = self._format_params(model.parameters)
            
            lines.append(f"   Training FLOP: {flop_str} ({confidence} confidence)")
            lines.append(f"   Parameters: {params_str} | Status: {model.status.value}")
            
            # Sources
            if model.sources:
                primary_source = model.sources[0]
                if "huggingface.co" in primary_source:
                    source_name = "Hugging Face"
                elif "anthropic.com" in primary_source:
                    source_name = "Anthropic"
                elif "ai.meta.com" in primary_source:
                    source_name = "Meta AI"
                else:
                    source_name = "External source"
                
                lines.append(f"   Source: {source_name}")
            
            # Optional reasoning
            if show_reasoning and model.reasoning:
                reasoning = model.reasoning[:100] + "..." if len(model.reasoning) > 100 else model.reasoning
                lines.append(f"   Reasoning: {reasoning}")
            
            lines.append("")  # Empty line between models
        
        return "\n".join(lines)


class FullFormatter(BaseFormatter):
    """Full formatter for displaying complete model data."""
    
    def format_models(self, models: List[Model], **kwargs) -> str:
        """Format models in full JSON view."""
        if not models:
            return json.dumps({"models": [], "count": 0}, indent=2)
        
        formatted_models = []
        
        for i, model in enumerate(models, 1):
            formatted_model = {
                "rank": i,
                "name": model.name,
                "developer": model.developer,
                "release_date": self._format_date(model.release_date),
                "parameters": model.parameters,
                "parameters_formatted": self._format_params(model.parameters),
                "training_flop": model.training_flop,
                "training_flop_formatted": self._format_flop(model.training_flop),
                "training_flop_confidence": model.training_flop_confidence.value,
                "estimation_method": model.estimation_method.value,
                "inference_flop_per_token": model.inference_flop_per_token,
                "status": model.status.value,
                "context_length": model.context_length,
                "architecture": model.architecture,
                "reasoning": model.reasoning,
                "benchmarks": model.benchmarks,
                "sources": model.sources,
                "last_updated": model.last_updated.isoformat(),
                "metadata": model.metadata,
            }
            formatted_models.append(formatted_model)
        
        return json.dumps({
            "models": formatted_models,
            "count": len(formatted_models),
            "generated_at": datetime.utcnow().isoformat()
        }, indent=2)


class TableFormatter(BaseFormatter):
    """Table formatter for displaying models in a tabular format."""
    
    def format_models(self, models: List[Model], **kwargs) -> str:
        """Format models as a table."""
        if not models:
            return "No models found matching the criteria."
        
        # Prepare table data
        headers = ["Rank", "Name", "Developer", "FLOP", "Confidence", "Parameters", "Status"]
        rows = []
        
        for i, model in enumerate(models, 1):
            rows.append([
                str(i),
                model.name[:30] + "..." if len(model.name) > 30 else model.name,
                model.developer[:15] + "..." if len(model.developer) > 15 else model.developer,
                self._format_flop(model.training_flop),
                model.training_flop_confidence.value[:6],  # Truncate to fit
                self._format_params(model.parameters),
                model.status.value.replace("_", " ")[:15]
            ])
        
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        # Create table
        lines = []
        
        # Header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            lines.append(row_line)
        
        return "\n".join(lines)


class StatsFormatter(BaseFormatter):
    """Formatter for dataset statistics."""
    
    def format_stats(self, stats: Dict[str, Any]) -> str:
        """Format dataset statistics."""
        lines = [
            "=== Dataset Statistics ===\n",
            f"Total models: {stats['total_models']}",
            f"Above 1e25 FLOP threshold: {stats['above_threshold']}",
            f"Below threshold: {stats['below_threshold']}",
            f"With FLOP estimates: {stats['with_flop_estimates']}",
            f"With parameter counts: {stats['with_parameters']}",
            f"Unique developers: {stats['developers']}",
            f"Data files loaded: {stats['data_files']}",
            "",
            "Confidence distribution:",
        ]
        
        for conf, count in stats['confidence_distribution'].items():
            lines.append(f"  {conf}: {count}")
        
        return "\n".join(lines)