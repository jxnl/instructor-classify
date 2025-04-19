import os
import sys
import tempfile
import yaml
from unittest.mock import patch, MagicMock
import pytest
from typer.testing import CliRunner

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from instructor_classify.cli import app


@pytest.fixture
def mock_config_file():
    """Create a temporary configuration file for testing."""
    config = {
        "models": ["gpt-3.5-turbo"],
        "definition_path": os.path.join(current_dir, "intent_classification.yaml"),
        "eval_sets": [os.path.join(current_dir, "example_evalset.yaml")],
        "parallel_mode": "sync",
        "n_jobs": 2
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(config, temp)
        temp_path = temp.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@patch("instructor_classify.cli.UnifiedEvaluator")
def test_cli_eval_default(mock_evaluator, mock_config_file):
    """Test CLI eval command with default settings."""
    runner = CliRunner()
    
    # Set up mocks
    mock_instance = MagicMock()
    mock_evaluator.return_value = mock_instance
    
    result = runner.invoke(app, ["eval", "--config", mock_config_file])
    
    # Check exit code
    assert result.exit_code == 0
    
    # Verify evaluator was called with correct config
    mock_evaluator.assert_called_once_with(mock_config_file)
    
    # Verify methods were called
    mock_instance.prepare.assert_called_once()
    mock_instance.run.assert_called_once()


@patch("instructor_classify.cli.UnifiedEvaluator")
def test_cli_eval_with_mode_override(mock_evaluator, mock_config_file):
    """Test CLI eval command with mode override."""
    runner = CliRunner()
    
    # Set up mocks
    mock_instance = MagicMock()
    mock_evaluator.return_value = mock_instance
    
    result = runner.invoke(app, ["eval", "--config", mock_config_file, "--mode", "parallel"])
    
    # Check exit code
    assert result.exit_code == 0
    
    # Verify evaluator was called, but not with the original config path
    # Since the CLI creates a temporary config file with the overrides
    mock_evaluator.assert_called_once()
    
    # Check the config was not the original one
    called_args = mock_evaluator.call_args[0][0]
    assert called_args != mock_config_file
    
    # Check the temporary file had the parallelism setting updated
    # We can't check the temp file directly as it's already deleted,
    # but we can verify through the output
    assert "CLI overrides: parallel_mode=parallel" in result.stdout


@patch("instructor_classify.cli.UnifiedEvaluator")
def test_cli_eval_with_jobs_override(mock_evaluator, mock_config_file):
    """Test CLI eval command with jobs override."""
    runner = CliRunner()
    
    # Set up mocks
    mock_instance = MagicMock()
    mock_evaluator.return_value = mock_instance
    
    result = runner.invoke(app, ["eval", "--config", mock_config_file, "--jobs", "8"])
    
    # Check exit code
    assert result.exit_code == 0
    
    # Verify evaluator was called, but not with the original config path
    # Since the CLI creates a temporary config file with the overrides
    mock_evaluator.assert_called_once()
    
    # Check the temporary file had the jobs setting updated
    assert "CLI overrides" in result.stdout
    assert "n_jobs=8" in result.stdout


@patch("instructor_classify.cli.UnifiedEvaluator")
def test_cli_eval_with_both_overrides(mock_evaluator, mock_config_file):
    """Test CLI eval command with both mode and jobs overrides."""
    runner = CliRunner()
    
    # Set up mocks
    mock_instance = MagicMock()
    mock_evaluator.return_value = mock_instance
    
    result = runner.invoke(app, [
        "eval", 
        "--config", mock_config_file, 
        "--mode", "async", 
        "--jobs", "4"
    ])
    
    # Check exit code
    assert result.exit_code == 0
    
    # Verify overrides were applied
    assert "CLI overrides" in result.stdout
    assert "parallel_mode=async" in result.stdout
    assert "n_jobs=4" in result.stdout


@patch("instructor_classify.cli.UnifiedEvaluator")
def test_cli_eval_with_invalid_mode(mock_evaluator, mock_config_file):
    """Test CLI eval command with invalid mode."""
    runner = CliRunner()
    
    # Set up mocks
    mock_instance = MagicMock()
    mock_evaluator.return_value = mock_instance
    
    result = runner.invoke(app, [
        "eval", 
        "--config", mock_config_file, 
        "--mode", "invalid_mode"
    ])
    
    # Command should still succeed, as the invalid mode is handled by the evaluator
    assert result.exit_code == 0
    
    # Verify evaluator was called
    mock_evaluator.assert_called_once()


def test_cli_init():
    """Test the init command creates expected files."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        
        # Run the init command
        result = runner.invoke(app, ["init", "test_project"])
        
        # Check exit code
        assert result.exit_code == 0
        
        # Check the project directory was created
        assert os.path.exists(os.path.join(tmpdir, "test_project"))
        
        # Check key files were created
        assert os.path.exists(os.path.join(tmpdir, "test_project", "prompt.yaml"))
        assert os.path.exists(os.path.join(tmpdir, "test_project", "example.py"))
        assert os.path.exists(os.path.join(tmpdir, "test_project", "datasets"))


def test_cli_help():
    """Test the help command displays help text."""
    runner = CliRunner()
    
    result = runner.invoke(app, ["--help"])
    
    # Check exit code
    assert result.exit_code == 0
    
    # Check help text contains command descriptions
    assert "init" in result.stdout
    assert "eval" in result.stdout