"""
Reporting pipeline stage.

This module provides the pipeline stage for generating reports from evaluation results.
"""

from typing import Any, Dict, List
import os

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.eval_harness.reporting import ConsoleReporter, FileReporter


class ReportingStage(PipelineStage):
    """
    Pipeline stage for generating reports from evaluation results.
    """
    
    def __init__(self):
        """Initialize the reporting stage."""
        super().__init__(name="Reporting")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the reporting stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        config = context["config"]
        results = context["results"]
        analysis_results = context["analysis_results"]
        output_dir = config.output_dir
        
        # Create reporters based on configuration
        reporters = []
        reporters_to_use = set(config.reporters)
        
        if "console" in reporters_to_use:
            reporters.append(ConsoleReporter(detailed=True))
        
        if "file" in reporters_to_use:
            reporters.append(FileReporter(output_dir=output_dir))
        
        # TODO: Add visualization reporter if needed
        
        # Generate reports
        self.console.print("\n[bold cyan]Generating Reports...[/bold cyan]")
        
        for reporter in reporters:
            reporter.generate_report(results, analysis_results)
        
        # If cache was used, show statistics
        if "cache" in context and context["cache"]:
            cache = context["cache"]
            stats = cache.get_stats()
            
            self.console.print("\n[bold cyan]Cache Statistics[/bold cyan]")
            self.console.print(f"Hits: {stats['hits']}")
            self.console.print(f"Misses: {stats['misses']}")
            self.console.print(f"Hit Rate: {stats['hit_rate']:.2%}")
            self.console.print(f"Cache Size: {stats['cache_size']} items")
        
        # Mark successful completion
        context["completed"] = True
        
        return context