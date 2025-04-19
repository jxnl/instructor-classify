"""
Console reporter for evaluation results.

This module provides rich console output for evaluation results.
"""

from typing import Any, Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from datetime import datetime

from instructor_classify.eval_harness.base import Reporter, EvaluationResult


class ConsoleReporter(Reporter):
    """
    Console reporter that displays results using Rich formatting.
    """
    
    def __init__(self, detailed: bool = True):
        """
        Initialize the console reporter.
        
        Parameters
        ----------
        detailed : bool
            Whether to show detailed results
        """
        super().__init__()
        self.detailed = detailed
    
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
        self._display_summary(eval_results)
        
        if self.detailed:
            self._display_detailed_results(eval_results)
            self._display_analysis_results(analysis_results)
        
        self._display_final_recommendation(eval_results, analysis_results)
    
    def _display_summary(self, eval_results: Dict[str, Dict[str, EvaluationResult]]) -> None:
        """
        Display a summary of evaluation results.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        """
        # Create summary table
        summary_table = Table(title="Model Performance Summary")
        summary_table.add_column("Evaluation Set", style="cyan")
        
        # Add columns for each model
        models = list(eval_results.keys())
        for model in models:
            summary_table.add_column(model, style="green")
        
        # Add a "best model" column
        summary_table.add_column("Best Model", style="yellow")
        
        # Get all evaluation sets
        eval_sets = set()
        for model_results in eval_results.values():
            eval_sets.update(model_results.keys())
        
        # Add rows for each eval set
        for eval_set in sorted(eval_sets):
            row = [eval_set]
            
            # Get accuracy for each model
            accuracies = {}
            for model in models:
                if model in eval_results and eval_set in eval_results[model]:
                    accuracy = eval_results[model][eval_set].accuracy
                    accuracies[model] = accuracy
                    row.append(f"{accuracy:.2%}")
                else:
                    row.append("N/A")
            
            # Determine best model
            if accuracies:
                best_model = max(accuracies, key=accuracies.get)
                row.append(best_model)
            else:
                row.append("N/A")
            
            summary_table.add_row(*row)
        
        # Add a row for average accuracy
        avg_row = ["Weighted Average"]
        avg_accuracies = {}
        
        for model in models:
            if model not in eval_results:
                avg_row.append("N/A")
                continue
                
            total_examples = 0
            weighted_sum = 0
            
            for eval_set, result in eval_results[model].items():
                examples = result.total_examples
                total_examples += examples
                weighted_sum += result.accuracy * examples
            
            avg = weighted_sum / total_examples if total_examples > 0 else 0
            avg_accuracies[model] = avg
            avg_row.append(f"{avg:.2%}")
        
        # Add best model for average
        if avg_accuracies:
            best_model = max(avg_accuracies, key=avg_accuracies.get)
            avg_row.append(best_model)
        else:
            avg_row.append("N/A")
        
        summary_table.add_row(*avg_row)
        
        # Display summary table
        self.console.print("\n[bold green]Evaluation Summary[/bold green]")
        self.console.print(summary_table)
    
    def _display_detailed_results(self, eval_results: Dict[str, Dict[str, EvaluationResult]]) -> None:
        """
        Display detailed evaluation results.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        """
        self.console.print("\n[bold green]Detailed Results by Model and Evaluation Set[/bold green]")
        
        for model, model_results in eval_results.items():
            self.console.print(f"\n[bold cyan]Model:[/bold cyan] {model}")
            
            for eval_set, result in model_results.items():
                self._display_evaluation_result(result)
    
    def _display_evaluation_result(self, result: EvaluationResult) -> None:
        """
        Display a single evaluation result.
        
        Parameters
        ----------
        result : EvaluationResult
            The evaluation result to display
        """
        # Create a summary table
        summary_table = Table(title=f"Evaluation Results: {result.eval_set_name}")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Model", result.model)
        summary_table.add_row("Total Examples", str(result.total_examples))
        summary_table.add_row("Correct Predictions", str(result.correct_predictions))
        summary_table.add_row("Accuracy", f"{result.accuracy:.2%}")
        
        self.console.print(summary_table)
        
        # Create a detailed results table for incorrect predictions
        if result.predictions:
            errors_table = Table(title="Incorrect Predictions")
            errors_table.add_column("Text", style="white", no_wrap=False)
            errors_table.add_column("Expected", style="green")
            errors_table.add_column("Predicted", style="red")
            
            # Add rows for incorrect predictions
            for pred in result.predictions:
                if not pred.get("is_correct", False):
                    # Truncate long texts
                    text = pred["text"]
                    if len(text) > 80:
                        text = text[:77] + "..."
                    
                    # Format expected and predicted values
                    expected = str(pred["expected"])
                    predicted = str(pred["predicted"])
                    
                    errors_table.add_row(text, expected, predicted)
            
            # Only display the table if there are errors
            if errors_table.row_count > 0:
                self.console.print("\nIncorrect Predictions:")
                self.console.print(errors_table)
    
    def _display_analysis_results(self, analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Display analysis results.
        
        Parameters
        ----------
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        self.console.print("\n[bold green]Analysis Results[/bold green]")
        
        for analyzer, analyzer_results in analysis_results.items():
            self.console.print(f"\n[bold cyan]Analyzer:[/bold cyan] {analyzer}")
            
            # For now, we just show that analysis was performed
            # The actual display is handled by the analyzer's display_results method
            # which has already been called during the analysis phase
            self.console.print(f"Analysis completed. Results available in output directory.")
    
    def _display_final_recommendation(self, eval_results: Dict[str, Dict[str, EvaluationResult]], 
                                     analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Display final recommendation.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        # Calculate average accuracy per model
        avg_accuracies = {}
        
        for model, model_results in eval_results.items():
            total_examples = 0
            weighted_sum = 0
            
            for result in model_results.values():
                examples = result.total_examples
                total_examples += examples
                weighted_sum += result.accuracy * examples
            
            avg = weighted_sum / total_examples if total_examples > 0 else 0
            avg_accuracies[model] = avg
        
        # Determine best model by accuracy
        best_accuracy_model = max(avg_accuracies, key=avg_accuracies.get) if avg_accuracies else None
        
        # Get cost information if available
        cost_analysis = analysis_results.get("cost", {})
        best_efficiency_model = None
        best_efficiency = 0
        
        for model, model_analyses in cost_analysis.items():
            for eval_set, analysis in model_analyses.items():
                if "costs" in analysis and model in analysis["costs"]:
                    cost = analysis["costs"][model].get("total_cost")
                    if cost and cost > 0:
                        efficiency = avg_accuracies.get(model, 0) / cost
                        if efficiency > best_efficiency:
                            best_efficiency = efficiency
                            best_efficiency_model = model
        
        # Display recommendation
        recommendation = []
        if best_accuracy_model:
            recommendation.append(f"Best accuracy: {best_accuracy_model} ({avg_accuracies[best_accuracy_model]:.2%})")
        
        if best_efficiency_model:
            recommendation.append(f"Best efficiency: {best_efficiency_model} ({best_efficiency:.2%}/$ ratio)")
        
        if recommendation:
            self.console.print("\n[bold green]Recommendation[/bold green]")
            for rec in recommendation:
                self.console.print(f"- {rec}")
        
        self.console.print("\n[bold green]Evaluation completed successfully![/bold green]")