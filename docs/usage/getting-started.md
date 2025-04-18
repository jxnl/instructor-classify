# Getting Started

This guide will help you set up and use Instructor Classify for LLM-based text classification.

## Project Initialization

Create a new classification project:

```bash
instruct-classify init my_classifier
cd my_classifier
```

This creates a project with:
- `prompt.yaml`: Classification schema definition
- `example.py`: Example code for using the classifier
- `configs/`: Evaluation configurations
- `datasets/`: Example evaluation datasets

## Understanding the Key Files

### prompt.yaml

The `prompt.yaml` file defines your classification schema, which is used to classify the user's input. Its often recommended to use an LLM to generate this schema.

```yaml
system_message: |
  You are an expert classification system designed to analyse user inputs.
  
label_definitions:
  - label: question
    description: The user asks for information or clarification.
    examples:
      examples_positive:
        - "What is the capital of France?"
        - "How does this feature work?"
      examples_negative:
        - "Please book me a flight to Paris."
        - "I want to return this product."
  
  - label: request
    description: The user wants the assistant to perform an action.
    examples:
      examples_positive:
        - "Please book me a flight to Paris."
        - "Update my account settings."
      examples_negative:
        - "What is the capital of France?"
        - "I'm having a problem with my order."
```

Each label definition includes:
- `label`: The category name (automatically converted to lowercase)
- `description`: What this category represents
- `examples`: Optional positive and negative examples to guide the model

!!! note "These are all prompts to the LLM"

    The LLM will use these labels to classify the user's input. Changing the labels will change the behavior of the LLM.

### example.py

The `example.py` file shows basic usage:

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

## Basic Classification

### Single-Label Classification

```python
# Single-label prediction
result = classifier.predict("What is the weather today?")
print(f"Label: {result.label}") # -> "question"
```

### Multi-Label Classification

```python
# Multi-label prediction
result = classifier.predict_multi("Can you help me find flights to Paris and book a hotel?")
print(f"Labels: {result.labels}") # -> ["request", "question"]
```

### Batch Processing

```python
# Process multiple texts in parallel
texts = [
    "What is machine learning?",
    "Please book a flight to New York.",
    "Can you explain how to use this API?"
]

# Synchronous batch processing
results = classifier.batch_predict(texts)
for text, result in zip(texts, results):
    print(f"Text: '{text}' → Label: {result.label}")
```

## Asynchronous API

For high-throughput applications, use the `AsyncClassifier`:

```python
from instructor_classify.classify import AsyncClassifier
from openai import AsyncOpenAI
import asyncio

async def main():
    # Create async classifier
    client = instructor.from_openai_aclient(AsyncOpenAI())
    classifier = (
        AsyncClassifier(definition)
        .with_client(client)
        .with_model("gpt-4o")
    )
    
    # Make predictions
    result = await classifier.predict("What is machine learning?")
    print(result.label)
    
    # Batch processing with concurrency control
    results = await classifier.batch_predict(texts, n_jobs=10)
    for text, result in zip(texts, results):
        print(f"Text: '{text}' → Label: {result.label}")

asyncio.run(main())
```

## Working with Multiple LLM Providers

Instructor Classify works with any provider supported by Instructor:

### OpenAI

```python
from openai import OpenAI
client = instructor.from_openai(OpenAI())
```

### Anthropic

```python
from anthropic import Anthropic
client = instructor.from_anthropic(Anthropic())
```

### Google

```python
import google.generativeai as genai
client = instructor.from_gemini(genai)
```

## Customizing Your Classifier

1. **Add or Modify Labels**: Edit `prompt.yaml` to add new categories
2. **Improve Examples**: Add more diverse examples to improve classification
3. **Adjust System Message**: Customize the initial instructions
4. **Switch Models**: Try different models with the `.with_model()` method

## Running Evaluations

Test your classifier's performance:

```bash
instruct-classify eval --config configs/example.yaml
```

The evaluation generates a detailed report with metrics, visualizations, and insights into model performance.

## Next Steps

- Learn about the [Evaluation Framework](evaluation.md) for benchmarking
- Check the [Examples](examples.md) for advanced usage patterns
- Refer to the [API Reference](../api.md) for detailed documentation