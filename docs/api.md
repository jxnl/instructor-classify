# API Reference

## CLI Commands

### `instruct-classify init`

Initialize a new classifier project.

```bash
instruct-classify init [PROJECT_NAME]
```

Creates a new project directory with:
- `prompt.yaml`: Classification definition
- `example.py`: Example code for using the classifier
- `configs/`: Evaluation configurations
- `datasets/`: Example evaluation datasets

### `instruct-classify eval`

Run evaluation using the unified evaluation framework.

```bash
instruct-classify eval --config CONFIG_PATH [--mode MODE] [--jobs N] [--cache/--no-cache]
```

**Parameters:**
- `--config`, `-c`: Path to the evaluation configuration YAML file
- `--mode`, `-m`: Parallelism mode: 'sync', 'parallel', or 'async'
- `--jobs`, `-j`: Number of parallel jobs to run (default: 4)
- `--cache/--no-cache`: Enable or disable caching

## Core Classes

### `ClassificationDefinition`

```python
from instructor_classify.schema import ClassificationDefinition
```

The schema for defining classification labels and examples.

**Methods:**

- `from_yaml(yaml_path: str) -> ClassificationDefinition`: Load definition from YAML
- `get_classification_model() -> type[BaseModel]`: Get the single-label Pydantic model
- `get_multiclassification_model() -> type[BaseModel]`: Get the multi-label Pydantic model

**Example:**
```python
definition = ClassificationDefinition.from_yaml("prompt.yaml")
```

### `Classifier`

```python
from instructor_classify.classify import Classifier
```

Fluent API for single and multi-label classification with synchronous operations.

**Methods:**

- `with_client(client: instructor.Instructor) -> Classifier`: Attach an LLM client
- `with_model(model_name: str) -> Classifier`: Specify the model to use
- `predict(text: str) -> BaseModel`: Make a single-label prediction
- `predict_multi(text: str) -> BaseModel`: Make a multi-label prediction
- `batch_predict(texts: list[str], n_jobs: int | None = None) -> list[BaseModel]`: Process multiple texts in parallel
- `batch_predict_multi(texts: list[str], n_jobs: int | None = None) -> list[BaseModel]`: Multi-label batch processing
- `predict_with_completion(text: str) -> tuple[BaseModel, Any]`: Return both parsed model and raw completion
- `predict_multi_with_completion(text: str) -> tuple[BaseModel, Any]`: Multi-label variant with raw completion

**Example:**
```python
classifier = (
    Classifier(definition)
    .with_client(instructor.from_openai(OpenAI()))
    .with_model("gpt-3.5-turbo")
)
result = classifier.predict("What is machine learning?")
```

### `AsyncClassifier`

```python
from instructor_classify.classify import AsyncClassifier
```

Asynchronous variant of the Classifier API. All prediction methods are coroutines.

**Methods:**

Same as Classifier, but all prediction methods are async and must be awaited.

**Example:**
```python
classifier = (
    AsyncClassifier(definition)
    .with_client(instructor.from_openai_aclient(AsyncOpenAI()))
    .with_model("gpt-4o")
)
result = await classifier.predict("What is machine learning?")
results = await classifier.batch_predict(texts, n_jobs=10)
```

### `EvalSet`

```python
from instructor_classify.schema import EvalSet
```

Holds a set of examples for evaluating classifier performance.

**Methods:**

- `from_yaml(yaml_path: str) -> EvalSet`: Load from YAML file
- `validate_against_definition(definition: ClassificationDefinition) -> bool`: Validate labels match definition

**Example:**
```python
eval_set = EvalSet.from_yaml("datasets/evalset_single.yaml")
```

## Evaluation Framework

### Legacy Interface

#### `UnifiedEvaluator`

```python
from instructor_classify.eval_harness.unified_eval import UnifiedEvaluator
```

Legacy evaluation framework for testing model performance (Deprecated in favor of the modular architecture).

**Methods:**

- `__init__(config_path: str)`: Initialize with configuration file
- `prepare()`: Prepare evaluation resources
- `run()`: Execute evaluation and generate reports

**Example:**
```python
evaluator = UnifiedEvaluator("configs/eval_config.yaml")
evaluator.prepare()
evaluator.run()
```

### Modular Evaluation System

#### `EvaluationOrchestrator`

```python
from instructor_classify.eval_harness.orchestrator import EvaluationOrchestrator
```

Main entry point for the modular evaluation system.

**Methods:**

- `__init__(config_path: str)`: Initialize with configuration file
- `execute() -> bool`: Run the evaluation pipeline
- `get_results() -> Dict[str, Dict[str, EvaluationResult]]`: Get results by model and dataset
- `get_analysis_results() -> Dict[str, Dict[str, Dict[str, Any]]]`: Get analysis results

**Example:**
```python
orchestrator = EvaluationOrchestrator("configs/eval_config.yaml")
success = orchestrator.execute()

if success:
    results = orchestrator.get_results()
    for model_name, eval_results in results.items():
        for eval_set_name, result in eval_results.items():
            print(f"{model_name} on {eval_set_name}: {result.accuracy:.2%}")
```

#### `EvaluationConfig`

```python
from instructor_classify.eval_harness.config.evaluation_config import EvaluationConfig
```

Configuration for the evaluation process.

**Methods:**

- `from_file(config_path: str) -> EvaluationConfig`: Load from YAML file
- `create_with_overrides(**overrides) -> EvaluationConfig`: Create new config with overrides
- `save_to_file(file_path: str) -> None`: Save to YAML file
- `create_temp_file() -> str`: Create temporary file with this configuration

**Example:**
```python
# Load from file
config = EvaluationConfig.from_file("configs/eval_config.yaml")

# Create programmatically 
config = EvaluationConfig(
    models=["gpt-3.5-turbo"],
    definition_path="prompt.yaml",
    eval_sets=["datasets/evalset.yaml"],
    parallel_mode="parallel", 
    n_jobs=4
)

# Override settings
new_config = config.create_with_overrides(parallel_mode="async", n_jobs=8)
```

#### `ProcessingStrategy`

```python
from instructor_classify.eval_harness.base import ProcessingStrategy
from instructor_classify.eval_harness.processing_strategies import (
    SyncProcessingStrategy, ParallelProcessingStrategy, AsyncProcessingStrategy
)
```

Abstract base class and implementations for processing evaluation examples.

**Interface:**

- `process_batch(classifier, examples, is_multi=False) -> List[Dict]`: Process a batch of examples

**Example:**
```python
# Create a specific strategy
strategy = ParallelProcessingStrategy(n_jobs=8)

# Used internally by the pipeline
results = strategy.process_batch(classifier, examples, is_multi=False)
```

#### `DiskCache`

```python
from instructor_classify.eval_harness.caching import DiskCache
```

Persistent cache for evaluation results.

**Methods:**

- `__init__(cache_dir: str = ".cache", use_pickle: bool = False)`: Initialize the cache
- `get(key: str) -> Optional[Any]`: Get a value from the cache
- `set(key: str, value: Any) -> None`: Set a value in the cache
- `clear() -> None`: Clear all cached values
- `get_stats() -> Dict[str, Any]`: Get cache statistics
- `generate_key(model: str, text: str, is_multi: bool = False) -> str`: Generate a cache key

**Example:**
```python
# Create cache
cache = DiskCache(cache_dir=".eval_cache")

# Use it
cache.set("key", value)
value = cache.get("key")

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

#### `Pipeline` and `PipelineStage`

```python
from instructor_classify.eval_harness.base import Pipeline, PipelineStage
from instructor_classify.eval_harness.pipeline import (
    ConfigStage, LoadStage, ModelStage, ExecutionStage, AnalysisStage, ReportingStage
)
```

Pipeline architecture for composable evaluation stages.

**Pipeline Methods:**

- `__init__(stages: List[PipelineStage] = None)`: Initialize with optional stages
- `add_stage(stage: PipelineStage) -> None`: Add a stage
- `execute() -> Dict[str, Any]`: Execute all stages

**PipelineStage Interface:**

- `__init__(name: str)`: Initialize with stage name
- `execute(context: Dict[str, Any]) -> Dict[str, Any]`: Execute stage logic

**Example:**
```python
# Create a custom pipeline
pipeline = Pipeline()
pipeline.add_stage(ConfigStage("configs/eval_config.yaml"))
pipeline.add_stage(LoadStage())
pipeline.add_stage(CustomStage())  # Your custom stage
pipeline.add_stage(ExecutionStage())
pipeline.add_stage(ReportingStage())

# Execute it
result_context = pipeline.execute()
```

## YAML Configuration Schemas

### Classification Definition (`prompt.yaml`)

```yaml
system_message: |
  You are an expert classification system...
  
label_definitions:
  - label: string
    description: string
    examples:
      examples_positive:
        - string
        - string
      examples_negative:
        - string
        - string

classification_type: "single" # or "multi"
```

### Evaluation Set (`evalset.yaml`)

```yaml
name: "Evaluation Set Name"
description: "Description of the evaluation set"
classification_type: "single" # or "multi"
examples:
  - text: "Example text to classify"
    expected_label: "expected_label" # for single-label
  - text: "Another example"
    expected_labels: ["label1", "label2"] # for multi-label
```

### Evaluation Configuration (`config.yaml`)

```yaml
# Models to evaluate
models:
  - "gpt-3.5-turbo"
  - "gpt-4o-mini"

# Paths to definition and evaluation datasets
definition_path: "definitions/prompt.yaml"
eval_sets:
  - "datasets/evalset_multi.yaml"
  - "datasets/evalset_single.yaml"

# Processing configuration
parallel_mode: "parallel"  # Options: sync, parallel, async
n_jobs: 4                  # Number of workers

# Analysis parameters
bootstrap_samples: 1000
confidence_level: 0.95

# Caching configuration
use_cache: true
cache_dir: ".eval_cache"

# Analyzers to use 
analyzers:
  - bootstrap
  - cost
  - confusion

# Reporters to use
reporters:
  - console
  - file

# Output directory
output_dir: "results"
```

## Error Handling

The following exceptions may be raised:

- `FileNotFoundError`: When the specified files cannot be found
- `yaml.YAMLError`: When YAML parsing fails
- `ValueError`: For invalid configurations or parameters
- Provider-specific errors: When LLM API calls fail