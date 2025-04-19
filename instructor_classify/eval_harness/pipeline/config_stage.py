"""
Configuration loading pipeline stage.

This module provides the pipeline stage for loading and validating configuration.
"""

from typing import Any, Dict
from rich.panel import Panel

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.eval_harness.config import EvaluationConfig


class ConfigStage(PipelineStage):
    """
    Pipeline stage for loading and validating configuration.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration stage.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
        """
        super().__init__(name="Configuration Loading")
        self.config_path = config_path
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the configuration loading stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        self.console.print(f"[green]Loading configuration from {self.config_path}[/green]")
        
        # Load configuration
        config = EvaluationConfig.from_file(self.config_path)
        
        # Display configuration summary
        self.console.print(
            Panel(
                f"[bold]Evaluation Configuration[/bold]\n\n"
                f"Models: {', '.join(config.models)}\n"
                f"Definition Path: {config.definition_path}\n"
                f"Evaluation Sets: {len(config.eval_sets)}\n"
                f"Parallel Mode: {config.parallel_mode} with {config.n_jobs} workers\n"
                f"Caching: {'Enabled' if config.use_cache else 'Disabled'}\n"
                f"Output Directory: {config.output_dir}",
                title="[bold cyan]Configuration Summary[/bold cyan]",
                border_style="cyan"
            )
        )
        
        # Update context
        context["config"] = config
        context["base_dir"] = self.config_path.rsplit("/", 1)[0] if "/" in self.config_path else "."
        
        return context