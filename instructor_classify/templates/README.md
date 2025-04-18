# Instructor Classify Examples

This directory contains example tests and configurations for [Instructor Classify](https://github.com/jxnl/instructor-classify), a fluent, type-safe API for text classification built on top of [Instructor](https://github.com/jxnl/instructor/).

## Directory Structure

- `example.py`: Basic demonstration of the Classifier with OpenAI
- `prompt.yaml`: Classification definition with three labels (question, scheduling, coding)
- `configs/`: Configuration files for evaluation
  - `example.yaml`: Configuration for evaluating models with multiple evaluation sets
  - `example_results/`: Results from evaluation runs (gitignored)
- `datasets/`: Evaluation datasets
  - `evalset_single.yaml`: Complex classification evaluation set with challenging examples
  - `evalset_multi.yaml`: Custom classification evaluation set with examples from prompt.yaml

## Getting Started

1. Install the required packages:
   ```bash
   pip install instructor pydantic openai
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your-api-key
   ```

3. Run the basic example:
   ```bash
   python example.py
   ```

## Running Evaluations

The evaluation harness allows comparing different models across multiple evaluation sets:

```bash
# From the parent directory
instruct-classify eval --config ./config/example.yaml
```

## Recent Evaluation Results

A recent evaluation comparing `gpt-3.5-turbo` and `gpt-4o-mini` showed:

### Performance

| Model | Custom Eval Set | Complex Eval Set | Average |
|-------|----------------|------------------|---------|
| gpt-3.5-turbo | 90.91% | 93.33% | 92.31% |
| gpt-4o-mini | 90.91% | 100.00% | 96.15% |

### Cost Efficiency

| Model | Cost | Efficiency |
|-------|------|------------|
| gpt-3.5-turbo | $0.0093 | 9900.01%/$ |
| gpt-4o-mini | $0.0279 | 3442.12%/$ |

**Recommendation:**
- Best accuracy: gpt-4o-mini (96.15%)
- Best efficiency: gpt-3.5-turbo (9900.01%/$ ratio)

## Notes

- All evaluation results are stored in the gitignored `configs/example_results/` directory
- Custom evaluation sets can be created following the format in the `datasets/` directory