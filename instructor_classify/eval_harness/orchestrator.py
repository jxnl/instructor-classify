"""
Evaluation orchestrator.

This module provides the main orchestrator class that coordinates the evaluation
process using the pipeline architecture.
"""

from typing import Any, Dict
from rich.console import Console
from rich.panel import Panel

from instructor_classify.eval_harness.base import Pipeline
from instructor_classify.eval_harness.pipeline import (
    ConfigStage, LoadStage, ModelStage, ExecutionStage, AnalysisStage, ReportingStage
)
from instructor_classify.eval_harness.utils.analysis import CostLatencyAnalyzer


class EvaluationOrchestrator:
    """
    Orchestrator for the evaluation process.
    
    This class coordinates the evaluation pipeline and handles the overall
    execution flow.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the evaluation orchestrator.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
        """
        self.config_path = config_path
        self.console = Console()
        self.pipeline = self._create_pipeline()
        
        # To be set during execution
        self.results = None
        self.analysis_results = None
        self.completed = False
    
    def _create_pipeline(self) -> Pipeline:
        """
        Create the evaluation pipeline.
        
        Returns
        -------
        Pipeline
            The created pipeline
        """
        pipeline = Pipeline()
        
        # Create and add common analyzers that are used by multiple stages
        CostLatencyAnalyzer()
        
        # Set up context with common analyzers
        
        # Add pipeline stages
        pipeline.add_stage(ConfigStage(self.config_path))
        pipeline.add_stage(LoadStage())
        pipeline.add_stage(ModelStage())
        pipeline.add_stage(ExecutionStage())
        pipeline.add_stage(AnalysisStage())
        pipeline.add_stage(ReportingStage())
        
        return pipeline
    
    def execute(self) -> bool:
        """
        Execute the evaluation process.
        
        Returns
        -------
        bool
            True if execution was successful, False otherwise
        """
        try:
            self.console.print(
                Panel(
                    "Starting evaluation process",
                    title="[bold cyan]Instructor Classify Evaluation[/bold cyan]",
                    border_style="cyan"
                )
            )
            
            # Execute the pipeline
            result_context = self.pipeline.execute()
            
            # Store results for access by clients
            self.results = result_context.get("results", {})
            self.analysis_results = result_context.get("analysis_results", {})
            self.completed = result_context.get("completed", False)
            
            self.console.print(
                Panel(
                    "Evaluation completed successfully",
                    title="[bold green]Success[/bold green]",
                    border_style="green"
                )
            )
            
            return True
        
        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Evaluation cancelled by user.[/bold yellow]")
            return False
        
        except Exception as e:
            self.console.print(f"\n[bold red]Error during evaluation:[/bold red] {str(e)}")
            import traceback
            self.console.print(traceback.format_exc())
            return False
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the evaluation results.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Evaluation results by model and eval set
        """
        if not self.completed:
            self.console.print("[yellow]Warning: Evaluation not completed, results may be incomplete[/yellow]")
        
        return self.results
    
    def get_analysis_results(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get the analysis results.
        
        Returns
        -------
        Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        if not self.completed:
            self.console.print("[yellow]Warning: Evaluation not completed, results may be incomplete[/yellow]")
        
        return self.analysis_results