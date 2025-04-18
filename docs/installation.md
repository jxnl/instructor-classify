# Installation

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Installation Options

### From PyPI

```bash
pip install instructor-classify
```

### From Source

1. Clone the repository:
```bash
git clone https://github.com/jasonliu/instructor-classify.git
cd instructor-classify
```

2. Install in development mode:
```bash
pip install -e .
```

## Required Dependencies

The package automatically installs these core dependencies:
- `instructor>=0.3.0`: Foundation for LLM structured output
- `pydantic>=2.0.0`: Type validation and schema generation
- `typer>=0.9.0`: CLI interface
- `pyyaml>=6.0.0`: Configuration handling
- `numpy>=2.0.2`, `matplotlib>=3.9.4`, `seaborn>=0.13.2`: Visualization support
- `scikit-learn>=1.6.1`: Evaluation metrics and analysis

## LLM Provider Libraries

You'll need to install at least one LLM provider client:

```bash
# For OpenAI
pip install openai

# For Anthropic
pip install anthropic

# For Google
pip install google-generativeai

# For other providers supported by Instructor
# See the Instructor documentation
```

## API Keys

Set up your API key for your chosen LLM provider:

```bash
# For OpenAI
export OPENAI_API_KEY=your-api-key-here

# For Anthropic
export ANTHROPIC_API_KEY=your-api-key-here

# For Google
export GOOGLE_API_KEY=your-api-key-here
```

## Optional Dependencies

For the best experience, you may want to install:

```bash
# Progress bars
pip install tqdm

# Documentation
pip install mkdocs mkdocs-material
```

## Verifying Installation

To verify that the installation was successful, run:

```bash
instruct-classify --help
```

You should see the help message with available commands.