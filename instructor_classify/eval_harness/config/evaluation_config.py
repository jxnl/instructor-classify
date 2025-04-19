"""
Configuration management for the evaluation harness.

This module handles loading, validation, and management of evaluation configurations.
"""

import os
import yaml
import tempfile
from typing import List
from pydantic import BaseModel, Field, validator
from rich.console import Console

from instructor_classify.eval_harness.base import ConfigLoader


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run."""
    
    # Model configuration
    models: List[str] = Field(..., description="List of models to evaluate")
    
    # Path configuration
    definition_path: str = Field(..., description="Path to classification definition YAML")
    eval_sets: List[str] = Field(..., description="Paths to evaluation set YAML files")
    output_dir: str = Field("results", description="Directory to store results")
    
    # Analysis configuration
    bootstrap_samples: int = Field(1000, description="Number of bootstrap samples for confidence intervals")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")
    
    # Processing configuration
    parallel_mode: str = Field("sync", description="Processing mode: sync, parallel, or async")
    n_jobs: int = Field(4, description="Number of parallel jobs")
    
    # Caching configuration
    use_cache: bool = Field(True, description="Whether to use caching")
    cache_dir: str = Field(".eval_cache", description="Directory for disk cache")
    
    # Analyzers to use (defaults to all)
    analyzers: List[str] = Field(
        ["bootstrap", "cost", "confusion"],
        description="Analyzers to use"
    )
    
    # Reporters to use (defaults to all)
    reporters: List[str] = Field(
        ["console", "file", "visualization"],
        description="Reporters to use"
    )
    
    @validator("parallel_mode")
    def validate_parallel_mode(cls, v):
        """Validate parallel mode."""
        if v not in ["sync", "parallel", "async"]:
            console = Console()
            console.print(f"[yellow]Warning: Invalid parallel_mode '{v}', defaulting to 'sync'[/yellow]")
            return "sync"
        return v
    
    @validator("confidence_level")
    def validate_confidence_level(cls, v):
        """Validate confidence level."""
        if not 0 < v < 1:
            console = Console()
            console.print(f"[yellow]Warning: Invalid confidence_level '{v}', defaulting to 0.95[/yellow]")
            return 0.95
        return v
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra fields for user-defined config options
    
    @classmethod
    def from_file(cls, config_path: str) -> "EvaluationConfig":
        """
        Load configuration from a YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration YAML file
            
        Returns
        -------
        EvaluationConfig
            Loaded configuration
        """
        console = Console()
        
        try:
            config_data = ConfigLoader.load_yaml(config_path)
            
            # Resolve paths to make them absolute
            base_dir = os.path.dirname(os.path.abspath(config_path))
            
            if "definition_path" in config_data and not os.path.isabs(config_data["definition_path"]):
                config_data["definition_path"] = ConfigLoader.resolve_path(
                    base_dir, config_data["definition_path"]
                )
            
            if "eval_sets" in config_data:
                for i, eval_set in enumerate(config_data["eval_sets"]):
                    if not os.path.isabs(eval_set):
                        config_data["eval_sets"][i] = ConfigLoader.resolve_path(
                            base_dir, eval_set
                        )
            
            return cls(**config_data)
        except Exception as e:
            console.print(f"[bold red]Error loading configuration from {config_path}:[/bold red] {str(e)}")
            raise
    
    def create_with_overrides(self, **overrides) -> "EvaluationConfig":
        """
        Create a new configuration with overrides.
        
        Parameters
        ----------
        **overrides
            Overrides for configuration values
            
        Returns
        -------
        EvaluationConfig
            New configuration with overrides applied
        """
        # Convert to dict, apply overrides, and create new instance
        config_dict = self.dict()
        config_dict.update(overrides)
        return EvaluationConfig(**config_dict)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Parameters
        ----------
        file_path : str
            Path to save configuration to
        """
        with open(file_path, 'w') as f:
            yaml.dump(self.dict(), f)
    
    def create_temp_file(self) -> str:
        """
        Create a temporary file with this configuration.
        
        Returns
        -------
        str
            Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            yaml.dump(self.dict(), temp)
            return temp.name