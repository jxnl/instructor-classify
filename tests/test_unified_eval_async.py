import os
import sys
import yaml
import tempfile
import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from instructor_classify.schema import ClassificationDefinition  # noqa: E402
from instructor_classify.eval_harness.unified_eval import UnifiedEvaluator  # noqa: E402


@pytest.fixture
def mock_async_config():
    """Create a temporary config file for async testing."""
    config_content = {
        "models": ["gpt-3.5-turbo"],
        "definition_path": os.path.join(current_dir, "intent_classification.yaml"),
        "eval_sets": [os.path.join(current_dir, "example_evalset.yaml")],
        "bootstrap_samples": 100,
        "confidence_level": 0.95,
        "n_jobs": 2,
        "parallel_mode": "async",
        "output_dir": "test_async_results"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(config_content, temp)
        temp_path = temp.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.mark.asyncio
async def test_process_batch_async():
    """Test the _process_batch_async method."""
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)
    
    # Create a simplified evaluator for testing
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.definition = definition
    evaluator.n_jobs = 2
    
    # Create a mock async classifier and examples
    class MockAsyncClassifier:
        async def predict(self, text):
            return MagicMock(label="question")
        
        async def predict_multi(self, text):
            return MagicMock(labels=["question"])
    
    class MockExample:
        def __init__(self, text, label=None, labels=None):
            self.text = text
            self.expected_label = label
            self.expected_labels = labels
    
    # Create examples
    examples = [
        MockExample(f"Example {i}", label="question") for i in range(5)
    ]
    
    # Mock asyncio semaphore
    class MockSemaphore:
        async def __aenter__(self):
            return None
        
        async def __aexit__(self, *args):
            return None
    
    # Patch asyncio.Semaphore to use our mock
    with patch('asyncio.Semaphore', return_value=MockSemaphore()):
        # Patch tqdm.gather to just run asyncio.gather
        with patch('tqdm.asyncio.tqdm.gather', new=asyncio.gather):
            results = await evaluator._process_batch_async(
                MockAsyncClassifier(), examples, is_multi=False
            )
            
            # Should have one result per example
            assert len(results) == len(examples)
            
            # Each result should have the expected structure
            for i, result in enumerate(results):
                assert result["text"] == f"Example {i}"
                assert result["is_correct"] is True
                assert result["expected"] == "question"
                assert result["predicted"] == "question"


def test_async_initialization(mock_async_config):
    """Test initialization with async mode."""
    evaluator = UnifiedEvaluator(mock_async_config)
    evaluator.prepare()
    
    assert evaluator.parallel_mode == "async"
    assert evaluator.n_jobs == 2
    assert len(evaluator.models) == 1
    assert evaluator.models[0] == "gpt-3.5-turbo"


@patch("asyncio.run")
@patch("instructor_classify.eval_harness.unified_eval.UnifiedEvaluator._process_batch_async")
def test_async_processing_setup(mock_process_async, mock_asyncio_run, mock_async_config):
    """Test setup for async processing."""
    # Create a mock async result
    mock_asyncio_run.return_value = [
        {
            "prediction": MagicMock(label="question"),
            "is_correct": True,
            "text": "Example text",
            "expected": "question",
            "predicted": "question"
        }
    ]
    
    # Initialize the evaluator
    evaluator = UnifiedEvaluator(mock_async_config)
    
    # Check it's configured for async
    assert evaluator.parallel_mode == "async"
    assert evaluator.n_jobs == 2


@pytest.mark.asyncio
async def test_async_process_with_different_job_counts():
    """Test async processing with different job counts."""
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)
    
    for n_jobs in [1, 2, 5]:
        # Create a simplified evaluator for testing
        evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
        evaluator.definition = definition
        evaluator.n_jobs = n_jobs
        
        # Create a mock async classifier and examples
        class MockAsyncClassifier:
            async def predict(self, text):
                await asyncio.sleep(0.01)  # Simulate network delay
                return MagicMock(label="question")
        
        class MockExample:
            def __init__(self, text, label=None):
                self.text = text
                self.expected_label = label
        
        # Create 10 examples
        examples = [
            MockExample(f"Example {i}", label="question") for i in range(10)
        ]
        
        # Mock tqdm.gather to just run asyncio.gather
        with patch('tqdm.asyncio.tqdm.gather', new=asyncio.gather):
            start_time = asyncio.get_event_loop().time()
            results = await evaluator._process_batch_async(
                MockAsyncClassifier(), examples, is_multi=False
            )
            end_time = asyncio.get_event_loop().time()
            
            # Should have processed all examples
            assert len(results) == 10
            
            # All should be correct
            assert all(r["is_correct"] for r in results)
            
            # Store execution time for comparison
            exec_time = end_time - start_time
            
            # With more jobs, execution should be faster (or at least not slower)
            # This is a timing test, so it's not deterministic
            # We're mainly checking that all examples are processed correctly
            print(f"Async processing with {n_jobs} jobs took {exec_time:.4f} seconds")


@pytest.mark.asyncio
async def test_multi_label_async_processing():
    """Test async processing for multi-label classification."""
    yaml_path = os.path.join(current_dir, "intent_classification.yaml")
    definition = ClassificationDefinition.from_yaml(yaml_path)
    
    # Create a simplified evaluator for testing
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.definition = definition
    evaluator.n_jobs = 2
    
    # Create a mock async classifier and examples
    class MockAsyncClassifier:
        async def predict_multi(self, text):
            if "wrong" in text:
                return MagicMock(labels=["coding"])
            return MagicMock(labels=["question"])
    
    class MockExample:
        def __init__(self, text, labels=None):
            self.text = text
            self.expected_labels = labels
    
    # Create examples - some correct, some incorrect
    examples = [
        MockExample("Example", labels=["question"]),
        MockExample("Example wrong", labels=["question"]),
    ]
    
    # Mock tqdm.gather to just run asyncio.gather
    with patch('tqdm.asyncio.tqdm.gather', new=asyncio.gather):
        results = await evaluator._process_batch_async(
            MockAsyncClassifier(), examples, is_multi=True
        )
        
        # Should have one result per example
        assert len(results) == 2
        
        # First example should be correct
        assert results[0]["is_correct"] is True
        assert results[0]["predicted"] == ["question"]
        
        # Second example should be incorrect
        assert results[1]["is_correct"] is False
        assert results[1]["predicted"] == ["coding"]