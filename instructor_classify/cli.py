import typer
from pathlib import Path
import shutil
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from instructor_classify.eval_harness.unified_eval import UnifiedEvaluator

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
    )
):
    """Run evaluation using the unified evaluation framework."""
    try:
        import yaml
        import tempfile
        import os as os_module  # Import with a different name to avoid shadowing
        
        # If CLI options are provided, update the config file
        if parallel_mode is not None or n_jobs is not None:
            # Load existing config
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Update with CLI options
            if parallel_mode is not None:
                config["parallel_mode"] = parallel_mode
            if n_jobs is not None:
                config["n_jobs"] = n_jobs
            
            # Fix paths to be absolute before writing to a new location
            config_dir = os_module.path.dirname(os_module.path.abspath(config_path))
            
            # Convert relative paths to absolute paths
            if "definition_path" in config and not os_module.path.isabs(config["definition_path"]):
                config["definition_path"] = os_module.path.normpath(os_module.path.join(config_dir, config["definition_path"]))
            
            if "eval_sets" in config:
                for i, eval_set in enumerate(config["eval_sets"]):
                    if not os_module.path.isabs(eval_set):
                        config["eval_sets"][i] = os_module.path.normpath(os_module.path.join(config_dir, eval_set))
            
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp:
                yaml.dump(config, temp)
                temp_config_path = temp.name
            
            # Use the updated config
            config_to_use = temp_config_path
            typer.echo(f"Using configuration with CLI overrides: parallel_mode={parallel_mode}, n_jobs={n_jobs}")
        else:
            # Use the original config
            config_to_use = config_path
        
        # Initialize and run evaluator
        evaluator = UnifiedEvaluator(config_to_use)
        evaluator.prepare()
        evaluator.run()
        
        # Clean up temporary file if created
        if parallel_mode is not None or n_jobs is not None:
            os_module.unlink(temp_config_path)
            
        typer.echo("\n[bold green]Evaluation completed successfully![/bold green]")
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