import typer
from pathlib import Path
import shutil
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# These imports need to come after the sys.path modification
# but ruff will still raise E402 warnings, so we use noqa
from instructor_classify.eval_harness.orchestrator import EvaluationOrchestrator  # noqa: E402
from instructor_classify.eval_harness.config.evaluation_config import EvaluationConfig  # noqa: E402

app = typer.Typer(name="instruct-classify")

def get_template_path() -> Path:
    return Path(__file__).parent / "templates"

@app.command("init")
def init(
    project_name: str = typer.Option(
        ...,
        prompt="What would you like to name your project directory?",
        help="Name of the project directory to create"
    )
):
    """Initialize a new classifier project."""
    project_dir = Path(project_name)
    project_dir.mkdir(exist_ok=True)
    
    # Copy entire templates directory
    template_path = get_template_path()
    shutil.copytree(template_path, project_dir, dirs_exist_ok=True)
    
    typer.echo(f"Created classifier project in {project_dir}")
    typer.echo("Copied all template files and directories:")
    for item in project_dir.rglob("*"):
        if item.is_file():
            typer.echo(f"  - {item.relative_to(project_dir)}")

@app.command("eval")
def evaluate(
    config_path: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the evaluation configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    parallel_mode: str = typer.Option(
        None,
        "--mode",
        "-m",
        help="Parallelism mode: 'sync' (sequential), 'parallel' (thread-based), or 'async' (asyncio-based)",
    ),
    n_jobs: int = typer.Option(
        None,
        "--jobs",
        "-j",
        help="Number of parallel jobs to run (default: 4)",
        min=1,
        max=32,
    ),
    use_cache: bool = typer.Option(
        None,
        "--cache/--no-cache",
        help="Enable or disable caching (default: from config or True)",
    )
):
    """Run evaluation using the unified evaluation framework."""
    try:
        # If CLI options are provided, create temporary config with overrides
        config_to_use = config_path
        temp_config_path = None
        
        if any(option is not None for option in [parallel_mode, n_jobs, use_cache]):
            # Load original config through the EvaluationConfig class for validation
            original_config = EvaluationConfig.from_file(config_path)
            
            # Create a dict with only the overrides that are specified
            overrides = {}
            if parallel_mode is not None:
                overrides["parallel_mode"] = parallel_mode
            if n_jobs is not None:
                overrides["n_jobs"] = n_jobs  
            if use_cache is not None:
                overrides["use_cache"] = use_cache
                
            # Apply overrides
            updated_config = original_config.create_with_overrides(**overrides)
            
            # Create temporary config file
            temp_config_path = updated_config.create_temp_file()
            config_to_use = temp_config_path
            
            # Show override info
            override_msg_parts = []
            if parallel_mode is not None:
                override_msg_parts.append(f"parallel_mode={parallel_mode}")
            if n_jobs is not None:
                override_msg_parts.append(f"n_jobs={n_jobs}")
            if use_cache is not None:
                override_msg_parts.append(f"cache={'enabled' if use_cache else 'disabled'}")
                
            typer.echo(f"Using configuration with CLI overrides: {', '.join(override_msg_parts)}")
        
        # Initialize and run evaluator
        evaluator = EvaluationOrchestrator(config_to_use)
        success = evaluator.execute()
        
        # Clean up temporary file if created
        if temp_config_path:
            os.unlink(temp_config_path)
            
        if success:
            typer.echo("\n[bold green]Evaluation completed successfully![/bold green]")
        else:
            typer.echo("\n[bold red]Evaluation failed.[/bold red]")
            raise typer.Exit(1)
            
    except KeyboardInterrupt:
        typer.echo("\n[bold yellow]Evaluation cancelled by user.[/bold yellow]")
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"\n[bold red]Error during evaluation:[/bold red] {str(e)}")
        import traceback
        typer.echo(traceback.format_exc())
        raise typer.Exit(1)

if __name__ == "__main__":
    app()