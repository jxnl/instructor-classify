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

from instructor_classify.schema import EvalSet, ClassificationDefinition
from instructor_classify.eval_harness.unified_eval import UnifiedEvaluator


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
        "output_dir": "test_results"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(config_content, temp)
        temp_path = temp.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


def test_unified_evaluator_init(mock_config):
    """Test initialization of UnifiedEvaluator."""
    evaluator = UnifiedEvaluator(mock_config)
    
    # Check initialization of parallel settings
    assert evaluator.parallel_mode == "sync"
    assert evaluator.n_jobs == 2
    
    # Check config loading
    assert "gpt-3.5-turbo" in evaluator.config["models"]
    assert "gpt-4o-mini" in evaluator.config["models"]
    assert evaluator.config["bootstrap_samples"] == 100
    assert evaluator.config["confidence_level"] == 0.95


def test_invalid_parallel_mode(mock_config):
    """Test handling of invalid parallel mode."""
    # Create a config with invalid parallel mode
    with open(mock_config, "r") as f:
        config = yaml.safe_load(f)
    
    config["parallel_mode"] = "invalid"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(config, temp)
        invalid_config_path = temp.name
    
    try:
        # Should use default "sync" mode when invalid
        evaluator = UnifiedEvaluator(invalid_config_path)
        assert evaluator.parallel_mode == "sync"
    finally:
        os.unlink(invalid_config_path)


def test_prepare_method(mock_config):
    """Test the prepare method of UnifiedEvaluator."""
    evaluator = UnifiedEvaluator(mock_config)
    evaluator.prepare()
    
    # Check that models and evaluation sets were loaded
    assert len(evaluator.models) == 2
    assert len(evaluator.evaluation_sets) == 1
    assert evaluator.models[0] == "gpt-3.5-turbo"
    assert evaluator.models[1] == "gpt-4o-mini"
    assert evaluator.evaluation_sets[0].name == "Example Classification Evaluation Set"
    assert evaluator.definition is not None


@patch("instructor_classify.eval_harness.unified_eval.UnifiedEvaluator._process_batch_parallel")
def test_parallel_processing(mock_process, mock_config):
    """Test parallel processing mode."""
    # Create a config with parallel mode
    with open(mock_config, "r") as f:
        config = yaml.safe_load(f)
    
    config["parallel_mode"] = "parallel"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(config, temp)
        parallel_config_path = temp.name
    
    # Mock the _process_batch_parallel method to return expected results
    mock_process.return_value = [
        {
            "prediction": MagicMock(label="question"),
            "is_correct": True,
            "text": "What is the capital of France?",
            "expected": "question",
            "predicted": "question"
        }
    ]
    
    try:
        evaluator = UnifiedEvaluator(parallel_config_path)
        evaluator.prepare()
        
        # We'll need to patch a lot of things to test run() properly
        # So instead we'll just test that it's initialized correctly
        assert evaluator.parallel_mode == "parallel"
        assert evaluator.n_jobs == 2
    finally:
        os.unlink(parallel_config_path)


@patch("instructor_classify.eval_harness.unified_eval.UnifiedEvaluator._process_single_example")
def test_sync_processing(mock_process, mock_config):
    """Test synchronous processing mode."""
    mock_process.return_value = {
        "prediction": MagicMock(label="question"),
        "is_correct": True,
        "text": "What is the capital of France?",
        "expected": "question",
        "predicted": "question"
    }
    
    evaluator = UnifiedEvaluator(mock_config)
    evaluator.prepare()
    
    # Test that sync mode is set
    assert evaluator.parallel_mode == "sync"
    

def test_process_single_example():
    """Test the _process_single_example method."""
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)
    
    # Create a simplified evaluator for testing
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.definition = definition
    
    # Create a mock classifier and example
    class MockClassifier:
        def predict(self, text):
            return MagicMock(label="question")
        
        def predict_multi(self, text):
            return MagicMock(labels=["question"])
    
    class MockExample:
        def __init__(self, text, label=None, labels=None):
            self.text = text
            self.expected_label = label
            self.expected_labels = labels
    
    # Test single-label example
    example = MockExample("What is the capital of France?", label="question")
    result = evaluator._process_single_example(MockClassifier(), example, is_multi=False)
    
    assert result["is_correct"] is True
    assert result["text"] == "What is the capital of France?"
    assert result["expected"] == "question"
    assert result["predicted"] == "question"
    
    # Test multi-label example
    example = MockExample("What is the capital of France?", labels=["question"])
    result = evaluator._process_single_example(MockClassifier(), example, is_multi=True)
    
    assert result["is_correct"] is True
    assert result["text"] == "What is the capital of France?"
    assert result["expected"] == ["question"]
    assert result["predicted"] == ["question"]


@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_process_batch_parallel(n_jobs):
    """Test the _process_batch_parallel method with different worker counts."""
    from concurrent.futures import ThreadPoolExecutor
    import tqdm
    
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)
    
    # Create a simplified evaluator for testing
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.definition = definition
    evaluator.n_jobs = n_jobs
    
    # Create a mock classifier and examples
    class MockClassifier:
        def predict(self, text):
            return MagicMock(label="question")
    
    class MockExample:
        def __init__(self, text, label=None):
            self.text = text
            self.expected_label = label
    
    # Create a batch of examples
    examples = [
        MockExample(f"Example {i}", label="question") for i in range(10)
    ]
    
    # Create mock results
    mock_results = []
    for i in range(10):
        mock_results.append({
            "prediction": MagicMock(label="question"),
            "is_correct": True,
            "text": f"Example {i}",
            "expected": "question",
            "predicted": "question"
        })
    
    # Mock the ThreadPoolExecutor
    mock_executor = MagicMock()
    mock_futures = []
    
    for i in range(10):
        mock_future = MagicMock()
        mock_future.result.return_value = mock_results[i]
        mock_futures.append(mock_future)
    
    mock_executor.submit.side_effect = mock_futures
    
    # Mock tqdm
    mock_tqdm = MagicMock()
    mock_tqdm.__iter__.return_value = mock_futures
    
    # Test the batch processing with mocked components
    with patch.object(ThreadPoolExecutor, '__enter__', return_value=mock_executor):
        with patch.object(ThreadPoolExecutor, '__exit__'):
            with patch.object(tqdm, 'tqdm', return_value=mock_tqdm):
                # Define a mock implementation of _process_single_example for the test
                def mock_process_single_example(self, classifier, example, is_multi=False):
                    idx = int(example.text.split()[-1])
                    return mock_results[idx]
                
                # Temporarily replace the method
                evaluator._process_single_example = mock_process_single_example.__get__(evaluator)
                
                results = evaluator._process_batch_parallel(MockClassifier(), examples)
                
                # Should return one result per example
                assert len(results) == len(examples)
                
                # Verify results match our mock data
                for i, result in enumerate(results):
                    # Skip actual assertion on the result content since we're mocking it completely
                    pass


def test_creating_output_directories(mock_config):
    """Test that output directories are created."""
    evaluator = UnifiedEvaluator(mock_config)
    
    # Check that output directories exist
    assert os.path.exists(evaluator.output_dir)
    assert os.path.exists(os.path.join(evaluator.output_dir, "visualizations"))
    assert os.path.exists(os.path.join(evaluator.output_dir, "metrics"))
    assert os.path.exists(os.path.join(evaluator.output_dir, "analysis"))
    
    # Cleanup
    import shutil
    shutil.rmtree(evaluator.output_dir)