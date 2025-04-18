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
    )
):
    """Run evaluation using the unified evaluation framework."""
    try:
        # Initialize and run evaluator
        evaluator = UnifiedEvaluator(config_path)
        evaluator.prepare()
        evaluator.run()
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