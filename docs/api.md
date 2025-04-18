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
instruct-classify eval --config CONFIG_PATH
```

**Parameters:**
- `--config`, `-c`: Path to the evaluation configuration YAML file

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

### `UnifiedEvaluator`

```python
from instructor_classify.eval_harness.unified_eval import UnifiedEvaluator
```

Comprehensive evaluation framework for testing model performance.

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

# Evaluation datasets
eval_sets:
  - "datasets/evalset_multi.yaml"
  - "datasets/evalset_single.yaml"

# Analysis parameters
bootstrap_samples: 1000
confidence_level: 0.95
```

## Error Handling

The following exceptions may be raised:

- `FileNotFoundError`: When the specified files cannot be found
- `yaml.YAMLError`: When YAML parsing fails
- `ValueError`: For invalid configurations or parameters
- Provider-specific errors: When LLM API calls fail