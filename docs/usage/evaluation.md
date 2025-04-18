# Evaluation Framework

Instructor Classify includes a comprehensive evaluation framework for testing and comparing model performance on classification tasks.

## Overview

The evaluation framework provides:

- **Performance metrics** (accuracy, precision, recall, F1 score)
- **Statistical analysis** with bootstrap confidence intervals
- **Error analysis** with confusion matrices
- **Cost and latency tracking**
- **Visualizations** and detailed reports

## Running Evaluations

You can run evaluations using the CLI:

```bash
instruct-classify eval --config configs/example.yaml
```

## Configuration

The evaluation configuration is defined in a YAML file:

```yaml
# Models to evaluate, we will search all models
models:
  - "gpt-3.5-turbo"
  - "gpt-4o-mini"

# Evaluation datasets
# We can segment each one based on certain splits 
eval_sets:
  - "datasets/evalset_multi.yaml"
  - "datasets/evalset_single.yaml"

# Analysis parameters
bootstrap_samples: 1000
confidence_level: 0.95

# Optional parameters
output_dir: "results"  # Where to save results
verbose: true          # Show detailed progress
```

## Evaluation Datasets

Evaluation datasets are defined in YAML files:

```yaml
name: "Custom Classification Evaluation Set"
description: "A set of examples for testing intent classification"
classification_type: "single"  # or "multi"
examples:
  - text: "How do I reset my password?"
    expected_label: "account_question"
  - text: "I need to update my billing information."
    expected_label: "billing_request"
  - text: "What time do you close today?"
    expected_label: "general_question"
  # Add more examples...
```

For multi-label classification, use `expected_labels` instead:

```yaml
  - text: "I'm having trouble logging in and need to update my payment method."
    expected_labels: ["account_question", "billing_request"]
```

## Outputs

The evaluation framework generates a comprehensive set of outputs:

### 1. Summary Report

A text file with overall results:

```
# Evaluation Summary Report

Date: 2025-04-18 19:31:16

## Models Evaluated
- gpt-3.5-turbo
- gpt-4o-mini

## Datasets
- Complex Classification Evaluation Set
- Custom Classification Evaluation Set

## Overall Performance
              | gpt-3.5-turbo | gpt-4o-mini  |
--------------|--------------|--------------|
Accuracy      | 0.8500       | 0.9250       |
Macro F1      | 0.8479       | 0.9268       |
Avg. Latency  | 0.6521s      | 0.8752s      |
Cost (tokens) | 21,450       | 24,680       |

## Bootstrap Confidence Intervals (95%)
              | gpt-3.5-turbo     | gpt-4o-mini       |
--------------|------------------|-------------------|
Accuracy      | 0.8025 - 0.8975  | 0.8850 - 0.9650   |
```

### 2. Metrics Files

Detailed JSON files for each model and dataset:

```json
{
  "accuracy": 0.925,
  "macro_precision": 0.9325,
  "macro_recall": 0.9231,
  "macro_f1": 0.9268,
  "per_label_metrics": {
    "account_question": {
      "precision": 0.9545,
      "recall": 0.9545,
      "f1": 0.9545
    },
    "billing_request": {
      "precision": 0.9333,
      "recall": 0.9333,
      "f1": 0.9333
    },
    "general_question": {
      "precision": 0.9091,
      "recall": 0.8824,
      "f1": 0.8955
    }
  }
}
```

### 3. Visualizations

The framework generates various visualizations:

- **Confusion matrices** for each model/dataset
- **Error distribution** charts
- **Bootstrap confidence interval** visualizations
- **Cost and latency** comparisons

### 4. Cost and Latency Analysis

JSON files with detailed cost and latency data:

```json
{
  "models": {
    "gpt-3.5-turbo": {
      "total_tokens": 21450,
      "estimated_cost_usd": 0.0429,
      "avg_tokens_per_prediction": 214.5,
      "avg_latency_seconds": 0.6521
    },
    "gpt-4o-mini": {
      "total_tokens": 24680,
      "estimated_cost_usd": 0.0494,
      "avg_tokens_per_prediction": 246.8,
      "avg_latency_seconds": 0.8752
    }
  }
}
```

## Advanced Analysis

### Bootstrap Analysis

The evaluation framework uses bootstrap resampling to estimate confidence intervals:

```json
{
  "bootstrap_samples": 1000,
  "confidence_level": 0.95,
  "metrics": {
    "gpt-3.5-turbo": {
      "Complex Classification Evaluation Set": {
        "accuracy": {
          "mean": 0.85,
          "lower_bound": 0.8025,
          "upper_bound": 0.8975
        }
      }
    }
  }
}
```

### Confusion Matrix Analysis

Detailed confusion matrices help identify specific error patterns:

```json
{
  "gpt-3.5-turbo": {
    "Custom Classification Evaluation Set": {
      "account_question": {
        "account_question": 21,
        "billing_request": 1,
        "general_question": 0
      },
      "billing_request": {
        "account_question": 1,
        "billing_request": 14,
        "general_question": 0
      },
      "general_question": {
        "account_question": 0,
        "billing_request": 1,
        "general_question": 16
      }
    }
  }
}
```

## Programmatic Access to Results

You can access evaluation results programmatically:

```python
from instructor_classify.eval_harness.unified_eval import UnifiedEvaluator
import json

# Run evaluation
evaluator = UnifiedEvaluator("configs/example.yaml")
evaluator.prepare()
results = evaluator.run()

# Access results
for model_name, model_results in results.items():
    print(f"Model: {model_name}")
    for eval_set_name, metrics in model_results.items():
        print(f"  Dataset: {eval_set_name}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        # Check if this model is better than others with statistical significance
        bootstrap_data = evaluator.bootstrap_results["metrics"][model_name][eval_set_name]["accuracy"]
        print(f"  95% CI: [{bootstrap_data['lower_bound']:.4f}, {bootstrap_data['upper_bound']:.4f}]")
```

## Custom Evaluation Metrics

You can extend the evaluation framework with custom metrics:

```python
from instructor_classify.eval_harness.unified_eval import UnifiedEvaluator
from sklearn.metrics import matthews_corrcoef

class CustomEvaluator(UnifiedEvaluator):
    def calculate_metrics(self, true_labels, pred_labels):
        # Get standard metrics
        metrics = super().calculate_metrics(true_labels, pred_labels)
        
        # Add Matthews Correlation Coefficient
        mcc = matthews_corrcoef(true_labels, pred_labels)
        metrics["matthews_corrcoef"] = mcc
        
        return metrics

# Use custom evaluator
evaluator = CustomEvaluator("configs/example.yaml")
evaluator.prepare()
results = evaluator.run()
```