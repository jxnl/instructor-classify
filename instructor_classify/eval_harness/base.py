"""
Base classes and interfaces for the modular evaluation harness.

This module provides the foundational abstract classes and interfaces that
define the architecture of the evaluation system.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Union, Type
from pydantic import BaseModel, Field
from rich.console import Console
import os
import yaml
import json
from pathlib import Path

from instructor_classify.schema import ClassificationDefinition, EvalSet, EvalExample


class EvaluationResult(BaseModel):
    """Model for storing evaluation results."""
    model: str
    eval_set_name: str
    total_examples: int
    correct_predictions: int
    accuracy: float
    predictions: List[Dict[str, Any]]


T = TypeVar('T')
P = TypeVar('P')
R = TypeVar('R')


class ProcessingStrategy(Generic[T, P, R], ABC):
    """
    Abstract base class for processing strategies.
    
    This defines how examples are processed during evaluation, whether
    synchronously, using thread-based parallelism, or using asyncio.
    """
    
    def __init__(self, n_jobs: int = 4):
        """
        Initialize the processing strategy.
        
        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs to run
        """
        self.n_jobs = n_jobs
    
    @abstractmethod
    def process_batch(self, classifier: T, examples: List[P], is_multi: bool = False) -> List[R]:
        """
        Process a batch of examples.
        
        Parameters
        ----------
        classifier : T
            The classifier to use
        examples : List[P]
            The examples to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        List[R]
            The processing results
        """
        pass


class ProcessingStrategyFactory:
    """Factory for creating processing strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, n_jobs: int = 4) -> ProcessingStrategy:
        """
        Create a processing strategy of the specified type.
        
        Parameters
        ----------
        strategy_type : str
            Type of strategy to create ('sync', 'parallel', or 'async')
        n_jobs : int
            Number of parallel jobs to run
            
        Returns
        -------
        ProcessingStrategy
            The created strategy
        """
        # Import implementations here to avoid circular imports
        from instructor_classify.eval_harness.processing_strategies.sync_strategy import SyncProcessingStrategy
        from instructor_classify.eval_harness.processing_strategies.parallel_strategy import ParallelProcessingStrategy
        from instructor_classify.eval_harness.processing_strategies.async_strategy import AsyncProcessingStrategy
        
        if strategy_type == "sync":
            return SyncProcessingStrategy(n_jobs)
        elif strategy_type == "parallel":
            return ParallelProcessingStrategy(n_jobs)
        elif strategy_type == "async":
            return AsyncProcessingStrategy(n_jobs)
        else:
            # Default to sync if invalid type specified
            return SyncProcessingStrategy(n_jobs)


class Analyzer(ABC):
    """
    Abstract base class for analyzers.
    
    Analyzers perform statistical analysis on evaluation results.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.console = Console()
    
    @abstractmethod
    def analyze(self, eval_result: EvaluationResult) -> Dict[str, Any]:
        """
        Analyze evaluation results.
        
        Parameters
        ----------
        eval_result : EvaluationResult
            The evaluation result to analyze
            
        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        pass
    
    @abstractmethod
    def display_results(self, analysis_results: Dict[str, Any]) -> None:
        """
        Display analysis results.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            The analysis results to display
        """
        pass
    
    def save_results(self, analysis_results: Dict[str, Any], output_path: str) -> None:
        """
        Save analysis results to a file.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            The analysis results to save
        output_path : str
            Path to save results to
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)


class Reporter(ABC):
    """
    Abstract base class for reporters.
    
    Reporters generate reports and visualizations from evaluation results.
    """
    
    def __init__(self):
        """Initialize the reporter."""
        self.console = Console()
    
    @abstractmethod
    def generate_report(self, eval_results: Dict[str, Dict[str, EvaluationResult]], 
                       analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Generate a report from evaluation results.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        pass


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Defines a single stage in the evaluation pipeline.
    """
    
    def __init__(self, name: str):
        """
        Initialize the pipeline stage.
        
        Parameters
        ----------
        name : str
            Name of the pipeline stage
        """
        self.name = name
        self.console = Console()
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context containing data from previous stages
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        pass


class Pipeline:
    """
    Evaluation pipeline that executes stages in sequence.
    """
    
    def __init__(self, stages: List[PipelineStage] = None):
        """
        Initialize the pipeline.
        
        Parameters
        ----------
        stages : List[PipelineStage], optional
            Pipeline stages to execute
        """
        self.stages = stages or []
        self.console = Console()
    
    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.
        
        Parameters
        ----------
        stage : PipelineStage
            Stage to add
        """
        self.stages.append(stage)
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute all stages in the pipeline.
        
        Returns
        -------
        Dict[str, Any]
            The final pipeline context
        """
        context = {}
        
        for stage in self.stages:
            self.console.print(f"[bold cyan]Executing pipeline stage:[/bold cyan] {stage.name}")
            try:
                context = stage.execute(context)
            except Exception as e:
                self.console.print(f"[bold red]Error in pipeline stage {stage.name}:[/bold red] {str(e)}")
                import traceback
                self.console.print(traceback.format_exc())
                # Propagate the exception to the caller
                raise
        
        return context


class Cache(ABC):
    """
    Abstract base class for caches.
    
    Caches store and retrieve cached values to avoid redundant computation.
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Optional[Any]
            Cached value, or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the cache."""
        pass


class ConfigLoader:
    """Utility for loading and validating configuration files."""
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Parameters
        ----------
        file_path : str
            Path to the YAML file
            
        Returns
        -------
        Dict[str, Any]
            Loaded configuration
        """
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            console = Console()
            console.print(f"[bold red]Error loading YAML file {file_path}:[/bold red] {str(e)}")
            raise
    
    @staticmethod
    def resolve_path(base_dir: str, path: str) -> str:
        """
        Resolve a path relative to a base directory if it is not absolute.
        
        Parameters
        ----------
        base_dir : str
            Base directory
        path : str
            Path to resolve
            
        Returns
        -------
        str
            Resolved absolute path
        """
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(base_dir, path))