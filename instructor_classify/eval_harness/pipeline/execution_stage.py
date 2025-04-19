"""
Execution pipeline stage.

This module provides the pipeline stage for executing the evaluation.
"""

from typing import Any, Dict, List
from rich.panel import Panel

from instructor_classify.eval_harness.base import PipelineStage, EvaluationResult
from instructor_classify.eval_harness.processing_strategies import (
    SyncProcessingStrategy, ParallelProcessingStrategy, AsyncProcessingStrategy
)


class ExecutionStage(PipelineStage):
    """
    Pipeline stage for executing the evaluation.
    """
    
    def __init__(self):
        """Initialize the execution stage."""
        super().__init__(name="Evaluation Execution")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the evaluation stage.
        
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
        evaluation_sets = context["evaluation_sets"]
        classifiers = context["classifiers"]
        async_classifiers = context.get("async_classifiers", {})
        
        # Create appropriate processing strategy
        strategy = None
        if config.parallel_mode == "sync":
            strategy = SyncProcessingStrategy(n_jobs=1)
        elif config.parallel_mode == "parallel":
            strategy = ParallelProcessingStrategy(n_jobs=config.n_jobs)
        elif config.parallel_mode == "async":
            strategy = AsyncProcessingStrategy(n_jobs=config.n_jobs)
        else:
            self.console.print(
                f"[yellow]Warning: Unknown parallel mode '{config.parallel_mode}', defaulting to sync[/yellow]"
            )
            strategy = SyncProcessingStrategy(n_jobs=1)
        
        # Display evaluation plan
        self.console.print(
            Panel(
                f"[bold]Evaluation Plan[/bold]\n\n"
                f"Models: {', '.join(config.models)}\n"
                f"Evaluation Sets: {', '.join(es.name for es in evaluation_sets)}\n"
                f"Processing Strategy: {config.parallel_mode.capitalize()} with {config.n_jobs} workers\n"
                f"Output Directory: {config.output_dir}",
                title="[bold cyan]Evaluation Execution[/bold cyan]",
                border_style="cyan"
            )
        )
        
        # Initialize results storage
        results_by_model = {}
        
        # Execute evaluation for each model and eval set
        for model_name in config.models:
            self.console.print(f"\n[bold cyan]Evaluating model:[/bold cyan] {model_name}")
            
            # Initialize model-specific results
            results_by_model[model_name] = {}
            
            # Get the appropriate classifier
            classifier = classifiers[model_name]
            
            # For async mode, use the async classifier
            if config.parallel_mode == "async" and model_name in async_classifiers:
                classifier = async_classifiers[model_name]
            
            # Evaluate on each evaluation set
            for eval_set in evaluation_sets:
                self.console.print(f"  - Evaluating on: {eval_set.name}")
                
                # Filter for examples with expected_label/expected_labels
                is_multi = eval_set.classification_type == "multi"
                if not is_multi:
                    valid_examples = [ex for ex in eval_set.examples if ex.expected_label]
                else:
                    valid_examples = [ex for ex in eval_set.examples if ex.expected_labels]
                
                # Skip if no valid examples
                if not valid_examples:
                    self.console.print(f"    [yellow]No valid examples found for {eval_set.name}[/yellow]")
                    continue
                
                # Process examples using the selected strategy
                results = strategy.process_batch(classifier, valid_examples, is_multi)
                
                # Extract metrics
                correct_count = sum(1 for r in results if r["is_correct"])
                accuracy = correct_count / len(valid_examples) if valid_examples else 0
                
                # Extract prediction details
                prediction_details = [
                    {
                        "text": r["text"],
                        "expected": r["expected"],
                        "predicted": r["predicted"],
                        "is_correct": r["is_correct"]
                    }
                    for r in results
                ]
                
                # Build result
                result = EvaluationResult(
                    model=model_name,
                    eval_set_name=eval_set.name,
                    total_examples=len(valid_examples),
                    correct_predictions=correct_count,
                    accuracy=accuracy,
                    predictions=prediction_details
                )
                
                # Store result
                results_by_model[model_name][eval_set.name] = result
                
                # Display accuracy
                self.console.print(f"    [green]Accuracy: {accuracy:.2%}[/green]")
        
        # Store results in context
        context["results"] = results_by_model
        
        return context