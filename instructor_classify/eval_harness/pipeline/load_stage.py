"""
Resource loading pipeline stage.

This module provides the pipeline stage for loading classification definitions
and evaluation sets.
"""

import os
import sys
from typing import Any, Dict
from rich.panel import Panel

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.schema import ClassificationDefinition, EvalSet


class LoadStage(PipelineStage):
    """
    Pipeline stage for loading classification definitions and evaluation sets.
    """
    
    def __init__(self):
        """Initialize the loading stage."""
        super().__init__(name="Resource Loading")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the resource loading stage.
        
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
        
        # Load classification definition
        self.console.print(f"[green]Loading classification definition from {config.definition_path}[/green]")
        
        if not os.path.exists(config.definition_path):
            self.console.print(
                f"[bold red]Error: Classification definition file not found: {config.definition_path}[/bold red]"
            )
            self.console.print(
                "[yellow]Hint: Make sure the path is correct and accessible. If using relative paths with CLI options, consider using absolute paths.[/yellow]"
            )
            sys.exit(1)
        
        try:
            definition = ClassificationDefinition.from_yaml(config.definition_path)
            context["definition"] = definition
        except Exception as e:
            self.console.print(
                f"[bold red]Error loading classification definition:[/bold red] {str(e)}"
            )
            sys.exit(1)
        
        # Load evaluation sets
        self.console.print("[green]Loading evaluation sets...[/green]")
        evaluation_sets = []
        
        for eval_set_path in config.eval_sets:
            if not os.path.exists(eval_set_path):
                self.console.print(
                    f"[yellow]Warning: Evaluation set file not found: {eval_set_path}[/yellow]"
                )
                self.console.print(
                    "[yellow]Hint: Make sure the path is correct and accessible. If using relative paths with CLI options, consider using absolute paths.[/yellow]"
                )
                continue
            
            try:
                eval_set = EvalSet.from_yaml(eval_set_path)
                evaluation_sets.append(eval_set)
                self.console.print(f"  - Loaded: {eval_set.name}")
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to load {eval_set_path}: {str(e)}[/yellow]"
                )
        
        if not evaluation_sets:
            self.console.print(
                "[bold red]Error: No valid evaluation sets loaded[/bold red]"
            )
            sys.exit(1)
        
        context["evaluation_sets"] = evaluation_sets
        
        # Display loading summary
        self.console.print(
            Panel(
                f"[bold]Resources Loaded[/bold]\n\n"
                f"Classification Definition: {definition.label_definitions[0].label} and {len(definition.label_definitions) - 1} other labels\n"
                f"Evaluation Sets: {len(evaluation_sets)}\n"
                f"Total Examples: {sum(len(es.examples) for es in evaluation_sets)}",
                title="[bold cyan]Resources Summary[/bold cyan]",
                border_style="cyan"
            )
        )
        
        return context