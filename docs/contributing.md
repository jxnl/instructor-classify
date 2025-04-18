# Contributing to Instructor Classify

We welcome contributions to Instructor Classify! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/instructor-classify.git
   cd instructor-classify
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines

3. Run tests to ensure your changes don't break existing functionality:
   ```bash
   pytest
   ```

4. Update documentation if needed:
   ```bash
   mkdocs serve  # To preview documentation locally
   ```

5. Commit your changes with a descriptive commit message:
   ```bash
   git add .
   git commit -m "feat: Add new classification feature"
   ```

6. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request on GitHub

## Commit Message Guidelines

We use conventional commits for our commit messages:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code restructuring without changing functionality
- `test:` for adding or modifying tests
- `chore:` for routine tasks, dependency updates, etc.

Example: `feat: Add support for Claude models`

## Testing

All new features and bug fixes should include tests. We use pytest for testing:

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_classifier.py

# Run tests with coverage
pytest --cov=instructor_classify
```

## Documentation

Documentation is written in Markdown and built with MkDocs:

- All new features should be documented
- Update existing documentation as needed
- Include code examples to demonstrate usage
- Run `mkdocs serve` to preview documentation locally before submitting

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Ensure all tests pass
3. Update documentation as needed
4. Submit your pull request with a clear title and description
5. Respond to any feedback or requested changes

## Release Process

For maintainers:

1. Update version in `pyproject.toml` following [Semantic Versioning](https://semver.org/)
2. Update `CHANGELOG.md` with all notable changes
3. Create a new release on GitHub with release notes
4. The CI workflow will automatically publish to PyPI

## Adding New Features

When adding new features, consider:

1. **Backward Compatibility**: Ensure existing code continues to work
2. **Performance**: New features should not significantly degrade performance
3. **Testing**: Include comprehensive tests
4. **Documentation**: Add clear documentation with examples

## Questions and Support

If you have questions about contributing:

- Open an issue on GitHub
- Ask in the repository's discussion section
- Contact the maintainers directly

Thank you for contributing to Instructor Classify!