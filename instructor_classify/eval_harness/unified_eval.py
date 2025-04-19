#!/usr/bin/env python3
"""
Unified evaluation framework for classification models.

This script provides a comprehensive framework for:
1. Running evaluations on multiple models and datasets
2. Tracking and analyzing cost and latency metrics
3. Generating statistical confidence intervals
4. Analyzing error patterns and confusion matrices
5. Visualizing results and comparisons

Usage:
  python unified_eval.py --config configs/example.yaml
"""

import os
import sys
import argparse
import yaml
from typing import Any
from datetime import datetime
from rich.console import Console
from rich.table import Table
import warnings

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# These imports need to come after the sys.path modification
from instructor_classify.schema import ClassificationDefinition, EvalSet  # noqa: E402
from instructor_classify.eval_harness.orchestrator import EvaluationOrchestrator  # noqa: E402

# Import local utilities
from instructor_classify.eval_harness.utils.analysis import BootstrapAnalyzer, CostLatencyAnalyzer, ConfusionAnalyzer  # noqa: E402
from instructor_classify.eval_harness.base import EvaluationResult  # noqa: E402


# This class is maintained for backward compatibility
class UnifiedEvaluator:
    """
    Unified evaluation framework for classification models.

    This class handles:
    - Loading models, definitions, and evaluation sets
    - Running synchronous or asynchronous evaluations
    - Collecting metrics on accuracy, cost, and latency
    - Generating comprehensive analysis and visualizations
    
    DEPRECATED: This class is maintained for backward compatibility.
    Use the new modular architecture in `EvaluationOrchestrator` for new code.
    """

    def __init__(self, config_path: str):
        """
        Initialize the unified evaluator with a configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file
        """
        warnings.warn(
            "UnifiedEvaluator is deprecated and will be removed in a future version. "
            "Use EvaluationOrchestrator instead.",
            DeprecationWarning, 
            stacklevel=2
        )
        
        self.console = Console()
        self.orchestrator = EvaluationOrchestrator(config_path)
        
        # Load configuration
        self.console.print(f"[green]Loading configuration from {config_path}[/green]")
        self.config = self._load_config(config_path)

        # Set up paths
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.output_dir = os.path.join(
            self.base_dir, self.config.get("output_dir", "results")
        )

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analysis"), exist_ok=True)

        # Set up analyzers
        self.cost_analyzer = CostLatencyAnalyzer()
        self.bootstrap_analyzer = BootstrapAnalyzer(
            n_resamples=self.config.get("bootstrap_samples", 1000),
            confidence_level=self.config.get("confidence_level", 0.95),
        )
        self.confusion_analyzer = ConfusionAnalyzer()

        # Set up parallelism configuration
        self.parallel_mode = self.config.get("parallel_mode", "sync")  # Options: sync, parallel, async
        self.n_jobs = self.config.get("n_jobs", 4)
        
        if self.parallel_mode not in ["sync", "parallel", "async"]:
            self.console.print(
                f"[yellow]Warning: Invalid parallel_mode '{self.parallel_mode}', defaulting to 'sync'[/yellow]"
            )
            self.parallel_mode = "sync"
            
        self.console.print(f"[green]Using {self.parallel_mode} mode with {self.n_jobs} workers[/green]")

        # Generate timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Tracking variables
        self.results_by_model = {}
        self.accuracy_by_model = {}
        self.evaluation_sets = []
        self.models = []

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.console.print(
                f"[bold red]Error loading configuration:[/bold red] {str(e)}"
            )
            sys.exit(1)

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the base directory if not absolute."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    def prepare(self) -> None:
        """
        Prepare for evaluation by loading all necessary components.

        This includes:
        - Loading the classification definition
        - Loading all evaluation sets
        - Setting up model configurations
        """
        # Forward to new implementation
        # Load classification definition
        if "definition_path" not in self.config:
            self.console.print(
                "[bold red]Error: No definition_path specified in config file[/bold red]"
            )
            sys.exit(1)
            
        definition_path = self._resolve_path(self.config["definition_path"])
        self.console.print(
            f"[green]Loading classification definition from {definition_path}[/green]"
        )

        # Check if the file exists
        if not os.path.exists(definition_path):
            self.console.print(
                f"[bold red]Error: Classification definition file not found: {definition_path}[/bold red]"
            )
            self.console.print(
                "[yellow]Hint: Make sure the path is correct and accessible. If using relative paths with CLI options, consider using absolute paths.[/yellow]"
            )
            sys.exit(1)
            
        try:
            self.definition = ClassificationDefinition.from_yaml(definition_path)
        except Exception as e:
            self.console.print(
                f"[bold red]Error loading classification definition:[/bold red] {str(e)}"
            )
            sys.exit(1)

        # Load evaluation sets
        self.console.print("[green]Loading evaluation sets...[/green]")
        
        if "eval_sets" not in self.config or not self.config["eval_sets"]:
            self.console.print(
                "[bold red]Error: No eval_sets specified in config file[/bold red]"
            )
            sys.exit(1)

        for eval_set_path in self.config["eval_sets"]:
            path = self._resolve_path(eval_set_path)
            
            # Check if the file exists
            if not os.path.exists(path):
                self.console.print(
                    f"[yellow]Warning: Evaluation set file not found: {path}[/yellow]"
                )
                self.console.print(
                    "[yellow]Hint: Make sure the path is correct and accessible. If using relative paths with CLI options, consider using absolute paths.[/yellow]"
                )
                continue
                
            try:
                eval_set = EvalSet.from_yaml(path)
                self.evaluation_sets.append(eval_set)
                self.console.print(f"  - Loaded: {eval_set.name}")
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to load {path}: {str(e)}[/yellow]"
                )

        if not self.evaluation_sets:
            self.console.print(
                "[bold red]Error: No valid evaluation sets loaded[/bold red]"
            )
            sys.exit(1)

        # Set up models
        self.models = self.config["models"]
        if not self.models:
            self.console.print(
                "[bold red]Error: No models specified in configuration[/bold red]"
            )
            sys.exit(1)

        self.console.print(
            f"[green]Prepared {len(self.models)} models and {len(self.evaluation_sets)} evaluation sets[/green]"
        )

    def _process_single_example(self, classifier, example, is_multi=False):
        """Process a single example with appropriate prediction method and timing."""
        if is_multi:
            prediction = classifier.predict_multi(example.text)
            is_correct = set(prediction.labels) == set(example.expected_labels)
            expected = example.expected_labels
            predicted = prediction.labels
        else:
            prediction = classifier.predict(example.text)
            is_correct = prediction.label == example.expected_label
            expected = example.expected_label
            predicted = prediction.label
            
        return {
            "prediction": prediction,
            "is_correct": is_correct,
            "text": example.text,
            "expected": expected,
            "predicted": predicted
        }
        
    def _process_batch_parallel(self, classifier, examples, is_multi=False):
        """Process a batch of examples using thread-based parallelism."""
        from concurrent.futures import ThreadPoolExecutor
        import tqdm
        
        results = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self._process_single_example, classifier, example, is_multi)
                for example in examples
            ]
            
            for future in tqdm.tqdm(
                futures, total=len(futures), desc="Classifying", leave=False
            ):
                results.append(future.result())
                
        return results
        
    async def _process_batch_async(self, classifier, examples, is_multi=False):
        """Process a batch of examples using asyncio-based parallelism."""
        import asyncio
        from tqdm.asyncio import tqdm
        
        async def _process_async(example):
            # Create async semaphore to limit concurrency
            async with asyncio.Semaphore(self.n_jobs):
                if is_multi:
                    prediction = await classifier.predict_multi(example.text)
                    is_correct = set(prediction.labels) == set(example.expected_labels)
                    expected = example.expected_labels
                    predicted = prediction.labels
                else:
                    prediction = await classifier.predict(example.text)
                    is_correct = prediction.label == example.expected_label
                    expected = example.expected_label
                    predicted = prediction.label
                    
                return {
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "text": example.text,
                    "expected": expected,
                    "predicted": predicted
                }
        
        # Create tasks
        tasks = [_process_async(example) for example in examples]
        
        # Process with progress bar
        results = await tqdm.gather(*tasks, desc="Classifying")
        return results

    def run(self) -> None:
        """
        Run the evaluation process for all models and evaluation sets.
        
        DEPRECATED: This method is maintained for backward compatibility.
        The actual implementation now uses the new modular architecture.
        """
        # Forward to new implementation
        result = self.orchestrator.execute()
        
        if result:
            # Copy results back for backward compatibility
            self.results_by_model = self.orchestrator.get_results()
        
        # Return for backward compatibility, even though this doesn't actually return anything
        return


def evaluate_classifier(classifier: Any, eval_set: EvalSet) -> EvaluationResult:
    """
    Evaluate a classifier against an evaluation set.
    
    Parameters
    ----------
    classifier : Any
        The classifier to evaluate. Must have batch_predict() or batch_predict_multi() methods.
    eval_set : EvalSet
        The evaluation set to use.
        
    Returns
    -------
    EvaluationResult
        The evaluation results.
    """
    # Determine if single or multi-label classification
    is_multi = eval_set.classification_type == "multi"
    
    # Filter for examples with expected labels
    if is_multi:
        valid_examples = [ex for ex in eval_set.examples if ex.expected_labels]
        
        # Get predictions
        predictions = classifier.batch_predict_multi([ex.text for ex in valid_examples])
        
        # Compare predictions to expected labels
        correct_count = 0
        prediction_details = []
        
        for example, pred in zip(valid_examples, predictions):
            is_correct = set(pred.labels) == set(example.expected_labels)
            if is_correct:
                correct_count += 1
                
            prediction_details.append({
                "text": example.text,
                "expected": example.expected_labels,
                "predicted": pred.labels,
                "is_correct": is_correct
            })
    else:
        # Single-label classification
        valid_examples = [ex for ex in eval_set.examples if ex.expected_label]
        
        # Get predictions
        predictions = classifier.batch_predict([ex.text for ex in valid_examples])
        
        # Compare predictions to expected labels
        correct_count = 0
        prediction_details = []
        
        for example, pred in zip(valid_examples, predictions):
            is_correct = pred.label == example.expected_label
            if is_correct:
                correct_count += 1
                
            prediction_details.append({
                "text": example.text,
                "expected": example.expected_label,
                "predicted": pred.label,
                "is_correct": is_correct
            })
    
    # Calculate accuracy
    accuracy = correct_count / len(valid_examples) if valid_examples else 0
    
    # Create result object
    result = EvaluationResult(
        model=getattr(classifier, "model", "unknown"),
        eval_set_name=eval_set.name,
        total_examples=len(valid_examples),
        correct_predictions=correct_count,
        accuracy=accuracy,
        predictions=prediction_details
    )
    
    return result


def display_evaluation_results(result: EvaluationResult) -> None:
    """
    Display evaluation results in a formatted table.
    
    Parameters
    ----------
    result : EvaluationResult
        The evaluation results to display.
    """
    console = Console()
    
    # Create a summary table
    summary_table = Table(title=f"Evaluation Results: {result.eval_set_name}")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Model", result.model)
    summary_table.add_row("Total Examples", str(result.total_examples))
    summary_table.add_row("Correct Predictions", str(result.correct_predictions))
    summary_table.add_row("Accuracy", f"{result.accuracy:.2%}")
    
    console.print(summary_table)
    
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
            console.print("\nIncorrect Predictions:")
            console.print(errors_table)


def main():
    """Main entry point for the unified evaluation framework."""
    parser = argparse.ArgumentParser(
        description="Unified classification evaluation framework"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the evaluation configuration YAML file",
    )

    args = parser.parse_args()

    try:
        # Use the new orchestrator directly
        orchestrator = EvaluationOrchestrator(args.config)
        success = orchestrator.execute()
        
        if success:
            Console().print("\n[bold green]Evaluation completed successfully![/bold green]")
        else:
            Console().print("\n[bold red]Evaluation failed.[/bold red]")
            sys.exit(1)

    except KeyboardInterrupt:
        Console().print("\n[bold yellow]Evaluation cancelled by user.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        Console().print(f"\n[bold red]Error during evaluation:[/bold red] {str(e)}")
        import traceback

        Console().print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()