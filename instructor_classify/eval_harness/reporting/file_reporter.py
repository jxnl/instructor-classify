"""
File reporter for evaluation results.

This module provides file-based output for evaluation results.
"""

import json
from typing import Any, Dict
from datetime import datetime
from pathlib import Path

from instructor_classify.eval_harness.base import Reporter, EvaluationResult


class FileReporter(Reporter):
    """
    File reporter that saves results to disk.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the file reporter.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results to
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
    
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
        # Save evaluation results
        self._save_evaluation_results(eval_results)
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        # Generate and save summary report
        self._generate_summary_report(eval_results, analysis_results)
    
    def _save_evaluation_results(self, eval_results: Dict[str, Dict[str, EvaluationResult]]) -> None:
        """
        Save evaluation results to files.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        """
        metrics_dir = self.output_dir / "metrics"
        
        for model, model_results in eval_results.items():
            for eval_set, result in model_results.items():
                # Convert to dict for serialization
                result_dict = result.dict()
                
                # Save to file
                result_path = metrics_dir / f"{model}_{eval_set.replace(' ', '_')}_{self.timestamp}.json"
                with open(result_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                self.console.print(f"Saved evaluation results to {result_path}")
    
    def _save_analysis_results(self, analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Save analysis results to files.
        
        Parameters
        ----------
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        analysis_dir = self.output_dir / "analysis"
        
        for analyzer, analyzer_results in analysis_results.items():
            # Save to file
            analysis_path = analysis_dir / f"{analyzer}_{self.timestamp}.json"
            
            # Convert analysis results to a serializable format
            serializable_results = self._make_serializable(analyzer_results)
            
            with open(analysis_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.console.print(f"Saved analysis results to {analysis_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.
        
        Parameters
        ----------
        obj : Any
            Object to convert
            
        Returns
        -------
        Any
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert other types to string
            return str(obj)
    
    def _generate_summary_report(self, eval_results: Dict[str, Dict[str, EvaluationResult]], 
                              analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Generate and save a summary report.
        
        Parameters
        ----------
        eval_results : Dict[str, Dict[str, EvaluationResult]]
            Evaluation results by model and eval set
        analysis_results : Dict[str, Dict[str, Dict[str, Any]]]
            Analysis results by analyzer, model, and eval set
        """
        summary_path = self.output_dir / f"summary_report_{self.timestamp}.txt"
        
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
        
        with open(summary_path, 'w') as f:
            f.write("Classification Evaluation Summary\n")
            f.write("===============================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models: {', '.join(eval_results.keys())}\n")
            
            # Get all evaluation sets
            eval_sets = set()
            for model_results in eval_results.values():
                eval_sets.update(model_results.keys())
            
            f.write(f"Evaluation Sets: {', '.join(eval_sets)}\n\n")
            
            f.write("Performance Summary:\n")
            for model, model_results in eval_results.items():
                f.write(f"\n{model}:\n")
                
                for eval_set, result in model_results.items():
                    f.write(f"  {eval_set}: {result.accuracy:.2%}\n")
                
                f.write(f"  Average: {avg_accuracies.get(model, 0):.2%}\n")
            
            # Write cost analysis if available
            if cost_analysis:
                f.write("\nCost Analysis:\n")
                
                for model, model_analyses in cost_analysis.items():
                    for eval_set, analysis in model_analyses.items():
                        if "costs" in analysis and model in analysis["costs"]:
                            costs = analysis["costs"][model]
                            
                            f.write(f"\n{model}:\n")
                            f.write(f"  Input Tokens: {costs.get('input_tokens', 'N/A')}\n")
                            f.write(f"  Output Tokens: {costs.get('output_tokens', 'N/A')}\n")
                            
                            total_cost = costs.get("total_cost")
                            if total_cost:
                                f.write(f"  Total Cost: ${total_cost:.4f}\n")
                                
                                efficiency = avg_accuracies.get(model, 0) / total_cost if total_cost > 0 else 0
                                f.write(f"  Efficiency: {efficiency:.2%}/$ (higher is better)\n")
                                
                                if efficiency > best_efficiency:
                                    best_efficiency = efficiency
                                    best_efficiency_model = model
            
            # Write recommendation
            f.write("\nRecommendation:\n")
            
            if best_accuracy_model:
                f.write(f"- Best accuracy: {best_accuracy_model} ({avg_accuracies[best_accuracy_model]:.2%})\n")
            
            if best_efficiency_model:
                f.write(f"- Best efficiency: {best_efficiency_model} ({best_efficiency:.2%}/$ ratio)\n")
        
        self.console.print(f"Saved summary report to {summary_path}")