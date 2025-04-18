# Instructor Classify

A **fluent, type-safe API** for text classification built on top of [Instructor](https://github.com/jxnl/instructor/) with support for multiple LLM providers.

## Features

- 🏷️ **Simple label definitions** in YAML or Python
- 🔒 **Type-safe classification** using Pydantic
- 🌐 **Multi-provider support** (OpenAI, Anthropic, Google, etc.)
- 📊 **Complete evaluation framework** with metrics and visualizations
- ⚡ **Async/Sync APIs** for flexibility in different environments
- 💰 **Cost & latency tracking** for optimizing LLM usage
- 🧪 **Statistical analysis** with bootstrap confidence intervals
- 🚀 **CLI tools** for project setup and evaluation

## Overview

Instructor Classify provides a comprehensive framework for implementing, testing, and evaluating LLM-based text classification systems. It builds on the Instructor library to ensure type safety and structured outputs, while adding specific functionality for classification tasks.

The package supports both single-label and multi-label classification, with flexible APIs for synchronous and asynchronous operation, batch processing, and detailed performance evaluation.

## Installation

```bash
# Install the package
pip install instructor-classify

# Install your preferred LLM provider
pip install openai  # Or anthropic, google-generativeai, etc.
```

## Basic Usage

### Defining Classifications in Python

Create classification schemas directly in code:

```python
from instructor_classify import Classifier, ClassificationDefinition, LabelDefinition, Examples
import instructor
from openai import OpenAI

# Create label definitions
question_label = LabelDefinition(
    label="question",
    description="User is asking for information or clarification",
    examples=Examples(
        examples_positive=[
            "What is machine learning?",
            "How do I reset my password?"
        ],
        examples_negative=[
            "Please book me a flight",
            "I'd like to cancel my subscription"
        ]
    )
)

request_label = LabelDefinition(
    label="request",
    description="User wants the assistant to perform an action",
    examples=Examples(
        examples_positive=[
            "Please book me a flight to Paris",
            "Schedule a meeting for tomorrow at 2pm"
        ],
        examples_negative=[
            "What is the capital of France?",
            "Can you explain quantum physics?"
        ]
    )
)

# Create classification definition
definition = ClassificationDefinition(
    label_definitions=[question_label, request_label]
)

# Create classifier
client = instructor.from_openai(OpenAI())
classifier = (
    Classifier(definition)
    .with_client(client)
    .with_model("gpt-3.5-turbo")
)

# Make a prediction
result = classifier.predict("What is machine learning?")
print(result.label)  # -> "question"
```

### Defining Classifications in YAML

For better version control and reusability, you can define schemas in YAML:

```yaml
# prompt.yaml
label_definitions:
  - label: question
    description: User is asking for information or clarification
    examples:
      examples_positive:
        - "What is machine learning?"
        - "How do I reset my password?"
      examples_negative:
        - "Please book me a flight"
        - "I'd like to cancel my subscription"
  - label: request
    description: User wants the assistant to perform an action
    examples:
      examples_positive:
        - "Please book me a flight to Paris"
        - "Schedule a meeting for tomorrow at 2pm"
      examples_negative:
        - "What is the capital of France?"
        - "Can you explain quantum physics?"
```

Then load it in your code:

```python
from instructor_classify.classify import Classifier
from instructor_classify.schema import ClassificationDefinition
import instructor
from openai import OpenAI

# Load classification definition from YAML
definition = ClassificationDefinition.from_yaml("prompt.yaml")

# Create classifier
client = instructor.from_openai(OpenAI())
classifier = (
    Classifier(definition)
    .with_client(client)
    .with_model("gpt-3.5-turbo")
)

# Make a prediction
result = classifier.predict("What is machine learning?")
print(result.label)  # -> "question"
```

### Using the CLI

For quick project setup:

```bash
# Set up your API key
export OPENAI_API_KEY=your-api-key-here

# Initialize a new classification project
instruct-classify init my_classifier
cd my_classifier
```

This creates a project with:

- `prompt.yaml`: Sample classification schema definition
- `example.py`: Sample code for using the classifier
- `configs/`: Sample configurations
- `datasets/`: Sample evaluation datasets

This should be enough for you to get started testing models

## Documentation

- [Installation](installation.md)
- [Getting Started](usage/getting-started.md)
- [Programmatic Definition](usage/programmatic-definition.md)
- [Examples](usage/examples.md)
- [API Reference](api.md)
- [Evaluation Framework](usage/evaluation.md)
- [Contributing](contributing.md)