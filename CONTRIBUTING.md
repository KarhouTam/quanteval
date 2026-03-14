# Contributing to QuantEval

Thank you for helping improve QuantEval.

## Development setup

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install -e ".[dev]"
```

## Local checks

Please run these before opening a pull request:

```bash
ruff check .
pytest
python -m build
twine check dist/*
```

## Pull request expectations

- keep changes focused
- add or update tests for behavior changes
- update English and Chinese documentation when public behavior changes
- prefer small, reusable Python utilities over notebook-only logic

## Reporting issues

Include the following when possible:

- reproduction steps
- expected behavior
- actual behavior
- Python version and dependency context
- a minimal code sample or notebook cell sequence

## Notebook guidance

- avoid hard-coded local absolute paths
- keep outputs useful but not misleading
- prefer package imports over `sys.path` hacks whenever possible

## License

By contributing, you agree that your contributions will be released under the MIT License.
