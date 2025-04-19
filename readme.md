# Instructor Classify

[![Run Tests](https://github.com/jxnl/instructor-classify/actions/workflows/run-tests.yml/badge.svg)](https://github.com/jxnl/instructor-classify/actions/workflows/run-tests.yml)

A **fluent, type-safe API** for text classification built on top of [Instructor](https://github.com/jxnl/instructor/) with support for multiple LLM providers.

> Both synchronous and asynchronous APIs are fully supported with comprehensive evaluation tools.

## Overview

Instructor Classify provides:

- **Simple label definitions** in YAML or Python
- **Production-ready classification** with strong type safety
- **Multi-provider support** (OpenAI, Anthropic, Google, etc.)
- **Complete evaluation framework** with metrics, visualizations, and confidence intervals
- **Async/Sync APIs** for flexibility in different environments
- **Cost & latency tracking** for optimizing LLM usage

## Features

- **Type-safe classification API** powered by Pydantic
- **Flexible prediction modes**:
  - Single-label and multi-label classification
  - Batch processing with parallel execution
  - Raw completions for custom processing
- **Comprehensive evaluation**:
  - Accuracy, precision, recall, and F1 metrics
  - Bootstrap confidence intervals
  - Confusion matrix analysis
  - Error pattern identification
  - Visualization generation
- **CLI interface** for project initialization and evaluation

## Installation

```bash
git clone https://github.com/jxnl/instructor-classify.git
cd instructor-classify
uv sync
```

Required dependencies:
```bash
uv pip install instructor pydantic
uv pip install openai  # Or anthropic, google-generativeai, etc.
```

## Quick Start

### 1. Initialize a project

```bash
instruct-classify init my_classifier
```

This creates a new project with all necessary files:
- `prompt.yaml`: Classification definition
- `example.py`: Example code for using the classifier
- `configs/`: Evaluation configurations
- `datasets/`: Example evaluation datasets

### 2. Define your labels

```yaml
# prompt.yaml
system_message: |
  You are an expert classification system designed to analyse user inputs.
label_definitions:
  - label: question
    description: The user asks for information or clarification.
    examples:
      examples_positive:
        - "What is the capital of France?"
      examples_negative:
        - "Please book me a flight to Paris."
  - label: request
    description: The user wants the assistant to perform an action.
  # Add more labels...
```

### 3. Use the classifier

```python
from instructor_classify.classify import Classifier
from instructor_classify.schema import ClassificationDefinition
import instructor
from openai import OpenAI

# Load classification definition
definition = ClassificationDefinition.from_yaml("prompt.yaml")

# Create classifier
client = instructor.from_openai(OpenAI())
classifier = (
    Classifier(definition)
    .with_client(client)
    .with_model("gpt-3.5-turbo")
)

# Make predictions
result = classifier.predict("What is machine learning?")
print(result.label)  # -> "question"
```

### 4. Batch processing

```python
# Process multiple texts in parallel
texts = [
    "What is machine learning?",
    "Please book a flight to New York.",
    "Can you explain how to use this API?"
]

# Synchronous batch processing
results = classifier.batch_predict(texts)

# Asynchronous batch processing
async_classifier = AsyncClassifier(definition).with_client(async_client).with_model("gpt-4o")
results = await async_classifier.batch_predict(texts, n_jobs=10)
```

## Evaluation Framework

Evaluate model performance across datasets:

```bash
instruct-classify eval --config configs/example.yaml
```

Configuration file:
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

The evaluation framework generates:
- Performance metrics for each model/dataset
- Statistical significance analysis
- Cost and latency comparisons
- Visualizations for error analysis
- Detailed reports with actionable insights

## Project Structure

When you initialize a project with `instruct-classify init`, you get:

```
my_classifier/
├── prompt.yaml            # Classification definition
├── example.py             # Example usage
├── configs/
│   └── example.yaml       # Evaluation configuration
└── datasets/
    ├── evalset_multi.yaml # Multi-label evaluation dataset
    └── evalset_single.yaml # Single-label evaluation dataset
```

## Documentation

For detailed documentation, visit our [documentation site](https://github.com/jxnl/instructor-classify/docs) or run:

```bash
mkdocs serve
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Good first Issue:

- [ ] Enable Pytest Examples to print outputs into the markdown
- [ ] Cost tracking only works for openai, it does not use the instructor usage tracking standard, 