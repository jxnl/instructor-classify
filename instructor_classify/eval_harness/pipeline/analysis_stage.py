"""
Analysis pipeline stage.

This module provides the pipeline stage for analyzing evaluation results.
"""

import os
from typing import Any, Dict

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.eval_harness.utils.analysis import (
    BootstrapAnalyzer, CostLatencyAnalyzer, ConfusionAnalyzer
)


class AnalysisStage(PipelineStage):
    """
    Pipeline stage for analyzing evaluation results.
    """
    
    def __init__(self):
        """Initialize the analysis stage."""
        super().__init__(name="Analysis")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the analysis stage.
        
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
        results_by_model = context["results"]
        output_dir = config.output_dir
        
        # Initialize analyzers if not already in context
        bootstrap_analyzer = context.get("bootstrap_analyzer")
        if not bootstrap_analyzer:
            bootstrap_analyzer = BootstrapAnalyzer(
                n_resamples=config.bootstrap_samples,
                confidence_level=config.confidence_level
            )
            context["bootstrap_analyzer"] = bootstrap_analyzer
        
        cost_analyzer = context.get("cost_analyzer")
        if not cost_analyzer:
            cost_analyzer = CostLatencyAnalyzer()
            context["cost_analyzer"] = cost_analyzer
        
        confusion_analyzer = context.get("confusion_analyzer")
        if not confusion_analyzer:
            confusion_analyzer = ConfusionAnalyzer()
            context["confusion_analyzer"] = confusion_analyzer
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)
        
        # Run all analyses
        self.console.print("\n[bold cyan]Running Comprehensive Analysis...[/bold cyan]")
        
        # Initialize analysis results storage
        analysis_results = {
            "bootstrap": {},
            "cost": {},
            "confusion": {}
        }
        
        # Run the selected analyzers
        analyzers_to_run = set(config.analyzers)
        
        # Run cost and latency analysis
        if "cost" in analyzers_to_run:
            self.console.print("\n[bold green]Cost and Latency Analysis[/bold green]")
            self._run_cost_analysis(cost_analyzer, results_by_model, analysis_results, output_dir)
        
        # Run bootstrap analysis
        if "bootstrap" in analyzers_to_run:
            self.console.print("\n[bold green]Confidence Interval Analysis[/bold green]")
            self._run_bootstrap_analysis(bootstrap_analyzer, results_by_model, analysis_results, output_dir)
        
        # Run confusion analysis
        if "confusion" in analyzers_to_run:
            self.console.print("\n[bold green]Confusion and Error Pattern Analysis[/bold green]")
            self._run_confusion_analysis(confusion_analyzer, results_by_model, analysis_results, output_dir)
        
        # Store analysis results in context
        context["analysis_results"] = analysis_results
        
        return context
    
    def _run_cost_analysis(self, analyzer, results_by_model, analysis_results, output_dir):
        """Run cost and latency analysis."""
        # Calculate costs
        costs = analyzer.calculate_costs()
        latency_stats = analyzer.calculate_latency_stats()
        
        # Display results
        analyzer.display_cost_analysis(costs)
        analyzer.display_latency_analysis(latency_stats)
        
        # Generate visualizations
        viz_dir = os.path.join(output_dir, "visualizations")
        
        # Latency distribution
        latency_viz = os.path.join(viz_dir, "latency_distribution.png")
        analyzer.plot_latency_distribution(latency_viz)
        
        # Calculate average accuracy per model for cost efficiency
        avg_accuracies = {}
        for model, eval_results in results_by_model.items():
            total_examples = 0
            weighted_sum = 0
            for eval_name, result in eval_results.items():
                examples = result.total_examples
                total_examples += examples
                weighted_sum += result.accuracy * examples
            
            avg_accuracies[model] = weighted_sum / total_examples if total_examples > 0 else 0
        
        # Cost efficiency
        efficiency_viz = os.path.join(viz_dir, "cost_efficiency.png")
        analyzer.plot_cost_efficiency(costs, avg_accuracies, efficiency_viz)
        
        # Store analysis results
        for model in results_by_model.keys():
            if model not in analysis_results["cost"]:
                analysis_results["cost"][model] = {}
            
            for eval_name in results_by_model[model].keys():
                if eval_name not in analysis_results["cost"][model]:
                    analysis_results["cost"][model][eval_name] = {}
                
                analysis_results["cost"][model][eval_name] = {
                    "costs": costs,
                    "latency": latency_stats,
                    "avg_accuracy": avg_accuracies
                }
    
    def _run_bootstrap_analysis(self, analyzer, results_by_model, analysis_results, output_dir):
        """Run bootstrap analysis."""
        viz_dir = os.path.join(output_dir, "visualizations")
        
        for model_name, eval_results in results_by_model.items():
            if model_name not in analysis_results["bootstrap"]:
                analysis_results["bootstrap"][model_name] = {}
            
            for eval_name, result in eval_results.items():
                # Run bootstrap analysis
                eval_bootstrap = analyzer.analyze(result)
                analysis_results["bootstrap"][model_name][eval_name] = eval_bootstrap
                
                # Display results
                self.console.print(f"\n[bold]Model: {model_name}, Evaluation: {eval_name}[/bold]")
                analyzer.display_results(eval_bootstrap)
                
                # Generate visualizations
                bootstrap_viz = os.path.join(
                    viz_dir,
                    f"bootstrap_{model_name}_{eval_name.replace(' ', '_')}.png"
                )
                analyzer.plot_bootstrap_distributions(eval_bootstrap, bootstrap_viz)
    
    def _run_confusion_analysis(self, analyzer, results_by_model, analysis_results, output_dir):
        """Run confusion analysis."""
        viz_dir = os.path.join(output_dir, "visualizations")
        
        for model_name, eval_results in results_by_model.items():
            if model_name not in analysis_results["confusion"]:
                analysis_results["confusion"][model_name] = {}
            
            for eval_name, result in eval_results.items():
                # Run confusion analysis
                confusion_analysis = analyzer.analyze(result)
                analysis_results["confusion"][model_name][eval_name] = confusion_analysis
                
                if confusion_analysis:
                    # Display results
                    self.console.print(f"\n[bold]Model: {model_name}, Evaluation: {eval_name}[/bold]")
                    analyzer.display_results(confusion_analysis)
                    
                    # Generate visualizations
                    cm_viz = os.path.join(
                        viz_dir,
                        f"confusion_matrix_{model_name}_{eval_name.replace(' ', '_')}.png"
                    )
                    analyzer.plot_confusion_matrix(confusion_analysis, cm_viz)
                    
                    error_viz = os.path.join(
                        viz_dir,
                        f"error_distribution_{model_name}_{eval_name.replace(' ', '_')}.png"
                    )
                    analyzer.plot_error_distribution(confusion_analysis, error_viz)