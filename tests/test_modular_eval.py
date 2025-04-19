import os
import sys
import yaml
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from instructor_classify.eval_harness.orchestrator import EvaluationOrchestrator  # noqa: E402
from instructor_classify.eval_harness.config.evaluation_config import EvaluationConfig  # noqa: E402
from instructor_classify.eval_harness.pipeline import (  # noqa: E402
    ConfigStage, LoadStage, ExecutionStage
)
from instructor_classify.eval_harness.processing_strategies import (  # noqa: E402
    SyncProcessingStrategy, ParallelProcessingStrategy, AsyncProcessingStrategy
)
from instructor_classify.eval_harness.caching import DiskCache  # noqa: E402


@pytest.fixture
def mock_config():
    """Create a temporary config file for testing."""
    config_content = {
        "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "definition_path": os.path.join(current_dir, "intent_classification.yaml"),
        "eval_sets": [os.path.join(current_dir, "example_evalset.yaml")],
        "bootstrap_samples": 100,
        "confidence_level": 0.95,
        "n_jobs": 2,
        "parallel_mode": "sync",
        "output_dir": "test_results",
        "use_cache": True,
        "cache_dir": ".test_cache"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(config_content, temp)
        temp_path = temp.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


def test_evaluation_config_loading(mock_config):
    """Test loading evaluation configuration."""
    config = EvaluationConfig.from_file(mock_config)
    
    assert config.models == ["gpt-3.5-turbo", "gpt-4o-mini"]
    assert config.n_jobs == 2
    assert config.parallel_mode == "sync"
    assert config.bootstrap_samples == 100
    assert config.confidence_level == 0.95
    assert config.use_cache is True


def test_config_validation():
    """Test validation of configuration values."""
    # Test with invalid parallel mode
    config = EvaluationConfig(
        models=["gpt-3.5-turbo"],
        definition_path="test.yaml",
        eval_sets=["test_eval.yaml"],
        parallel_mode="invalid"
    )
    
    # Should be corrected to "sync"
    assert config.parallel_mode == "sync"
    
    # Test with invalid confidence level
    config = EvaluationConfig(
        models=["gpt-3.5-turbo"],
        definition_path="test.yaml",
        eval_sets=["test_eval.yaml"],
        confidence_level=1.5
    )
    
    # Should be corrected to 0.95
    assert config.confidence_level == 0.95


def test_config_overrides():
    """Test creating configurations with overrides."""
    base_config = EvaluationConfig(
        models=["gpt-3.5-turbo"],
        definition_path="test.yaml",
        eval_sets=["test_eval.yaml"],
        parallel_mode="sync",
        n_jobs=2
    )
    
    # Override parallel mode and jobs
    new_config = base_config.create_with_overrides(
        parallel_mode="parallel",
        n_jobs=4
    )
    
    assert new_config.parallel_mode == "parallel"
    assert new_config.n_jobs == 4
    assert new_config.models == ["gpt-3.5-turbo"]  # Unchanged
    
    # Add a new field
    new_config = base_config.create_with_overrides(
        use_cache=False
    )
    
    assert new_config.use_cache is False
    assert new_config.parallel_mode == "sync"  # Unchanged


def test_processing_strategies():
    """Test processing strategy implementations."""
    # Test SyncProcessingStrategy
    sync_strategy = SyncProcessingStrategy(n_jobs=1)
    assert sync_strategy.n_jobs == 1
    
    # Test ParallelProcessingStrategy
    parallel_strategy = ParallelProcessingStrategy(n_jobs=4)
    assert parallel_strategy.n_jobs == 4
    
    # Test AsyncProcessingStrategy
    async_strategy = AsyncProcessingStrategy(n_jobs=8)
    assert async_strategy.n_jobs == 8


def test_disk_cache():
    """Test disk cache implementation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskCache(cache_dir=temp_dir)
        
        # Test set and get
        cache.set("test_key", {"value": 42})
        result = cache.get("test_key")
        
        assert result is not None
        assert result["value"] == 42
        
        # Test generating prediction keys
        key1 = cache.generate_key("gpt-4", "Hello world", False)
        key2 = cache.generate_key("gpt-4", "Hello world", True)
        
        # Keys should be different for same text with different is_multi values
        assert key1 != key2
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["cache_size"] == 1
        
        # Test clear
        cache.clear()
        assert cache.get("test_key") is None
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["cache_size"] == 0


@patch("instructor_classify.eval_harness.pipeline.config_stage.EvaluationConfig.from_file")
def test_config_stage(mock_from_file, mock_config):
    """Test the configuration loading stage."""
    # Mock the config loading
    mock_config_obj = EvaluationConfig(
        models=["gpt-3.5-turbo"],
        definition_path="test.yaml",
        eval_sets=["test_eval.yaml"]
    )
    mock_from_file.return_value = mock_config_obj
    
    # Create and execute the stage
    stage = ConfigStage(mock_config)
    context = stage.execute({})
    
    # Check that the config was added to the context
    assert "config" in context
    assert context["config"] == mock_config_obj
    assert "base_dir" in context


@patch("os.path.exists")
@patch("instructor_classify.schema.ClassificationDefinition.from_yaml")
@patch("instructor_classify.schema.EvalSet.from_yaml")
def test_load_stage(mock_eval_from_yaml, mock_def_from_yaml, mock_exists):
    """Test the resource loading stage."""
    # Mock file existence
    mock_exists.return_value = True
    
    # Mock definition
    mock_definition = MagicMock()
    mock_definition.label_definitions = [MagicMock(label="test_label")]
    mock_def_from_yaml.return_value = mock_definition
    
    # Mock eval set
    mock_eval_set = MagicMock()
    mock_eval_set.name = "Test Eval Set"
    mock_eval_set.examples = [MagicMock(), MagicMock()]
    mock_eval_from_yaml.return_value = mock_eval_set
    
    # Create context with config
    context = {
        "config": EvaluationConfig(
            models=["gpt-3.5-turbo"],
            definition_path="test.yaml",
            eval_sets=["test_eval.yaml"]
        )
    }
    
    # Create and execute the stage
    stage = LoadStage()
    result_context = stage.execute(context)
    
    # Check results
    assert "definition" in result_context
    assert result_context["definition"] == mock_definition
    assert "evaluation_sets" in result_context
    assert len(result_context["evaluation_sets"]) == 1
    assert result_context["evaluation_sets"][0] == mock_eval_set


@patch("instructor_classify.eval_harness.pipeline.execution_stage.SyncProcessingStrategy")
def test_execution_stage(mock_strategy_class):
    """Test the execution stage."""
    # Mock strategy
    mock_strategy = MagicMock()
    mock_strategy.process_batch.return_value = [
        {
            "prediction": MagicMock(),
            "is_correct": True,
            "text": "test text",
            "expected": "expected",
            "predicted": "predicted"
        }
    ]
    mock_strategy_class.return_value = mock_strategy
    
    # Create mock classifier
    mock_classifier = MagicMock()
    
    # Create mock eval set
    mock_eval_set = MagicMock()
    mock_eval_set.name = "Test Eval Set"
    mock_eval_set.classification_type = "single"
    mock_eval_set.examples = [
        MagicMock(text="test text", expected_label="expected")
    ]
    
    # Create context
    context = {
        "config": EvaluationConfig(
            models=["test-model"],
            definition_path="test.yaml",
            eval_sets=["test_eval.yaml"],
            parallel_mode="sync"
        ),
        "evaluation_sets": [mock_eval_set],
        "classifiers": {"test-model": mock_classifier},
        "async_classifiers": {}
    }
    
    # Create and execute the stage
    stage = ExecutionStage()
    result_context = stage.execute(context)
    
    # Check results
    assert "results" in result_context
    assert "test-model" in result_context["results"]
    assert mock_eval_set.name in result_context["results"]["test-model"]
    assert result_context["results"]["test-model"][mock_eval_set.name].model == "test-model"
    assert result_context["results"]["test-model"][mock_eval_set.name].accuracy == 1.0


@patch("instructor_classify.eval_harness.orchestrator.EvaluationOrchestrator._create_pipeline")
def test_orchestrator_initialization(mock_create_pipeline, mock_config):
    """Test initialization of the evaluation orchestrator."""
    mock_pipeline = MagicMock()
    mock_create_pipeline.return_value = mock_pipeline
    
    orchestrator = EvaluationOrchestrator(mock_config)
    
    assert orchestrator.config_path == mock_config
    assert orchestrator.pipeline == mock_pipeline
    assert orchestrator.results is None
    assert orchestrator.analysis_results is None
    assert orchestrator.completed is False


@patch("instructor_classify.eval_harness.orchestrator.Pipeline.execute")
def test_orchestrator_execution(mock_execute, mock_config):
    """Test execution of the evaluation orchestrator."""
    # Mock pipeline execution result
    mock_results = {
        "test-model": {
            "Test Eval Set": MagicMock()
        }
    }
    mock_analysis_results = {
        "bootstrap": {},
        "cost": {},
        "confusion": {}
    }
    mock_execute.return_value = {
        "results": mock_results,
        "analysis_results": mock_analysis_results,
        "completed": True
    }
    
    # Create orchestrator and execute
    orchestrator = EvaluationOrchestrator(mock_config)
    success = orchestrator.execute()
    
    # Check results
    assert success is True
    assert orchestrator.results == mock_results
    assert orchestrator.analysis_results == mock_analysis_results
    assert orchestrator.completed is True
    
    # Test result getters
    assert orchestrator.get_results() == mock_results
    assert orchestrator.get_analysis_results() == mock_analysis_results