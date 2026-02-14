# Contributing Guide

Thank you for contributing to the VGGT Dataset Builder! This document outlines the standards and workflow for contributors.

## Code Style

This project uses **Black** for code formatting. All Python code must be formatted with Black before submission.

### Running Black

```bash
# Format all files
uv run black .

# Format specific files
uv run black dataset_utils.py build_warp_dataset.py

# Check without modifying
uv run black --check .

# See diffs
uv run black --diff .
```

### Black Configuration

- **Line length**: 88 characters
- **Python version**: 3.10+
- **String quotes**: Double quotes (unless single quotes require fewer escapes)
- **Trailing commas**: Added in multi-line structures

See `pyproject.toml` for full configuration.

## Pre-Commit Hooks (Recommended)

Set up automated formatting and checks:

```bash
# Install pre-commit
uv pip install pre-commit

# Install git hooks
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files
```

This will automatically:
- Format code with Black
- Sort imports with isort
- Check code style with flake8
- Fix trailing whitespace

## Pull Request Workflow

1. **Create a branch**: `git checkout -b feature/my-feature`
2. **Make changes**: Write your code
3. **Format code**: `uv run black .`
4. **Run tests**: `uv run pytest tests/ -v`
5. **Commit**: Use clear, descriptive commit messages
6. **Push and create PR**

## Testing

Always run tests before submitting:

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_triplet_detection.py -v

# Run with coverage
uv run pytest --cov=. tests/
```

## Linting & Quality

Optional additional checks:

```bash
# Lint with pylint
uv run pylint *.py

# Check formatting
uv run black --check .

# Full pre-commit check
uv run pre-commit run --all-files
```

## Using `uv run`

**Always use `uv run` when executing Python commands.** This ensures:
- Correct virtual environment is used
- Dependencies are properly isolated
- Consistent behavior across developers

❌ `python script.py`  
✅ `uv run python script.py`

## Commit Messages

Use clear, concise commit messages following this pattern:

```
[category] Brief description of changes

Longer explanation if needed:
- Point 1
- Point 2

Related issues/PRs: #123
```

### Categories

- 🏜️ `Dryer:` - Code deduplication/refactoring
- ✨ `Feature:` - New functionality
- 🐛 `Fix:` - Bug fixes
- 📝 `Docs:` - Documentation updates
- ♻️ `Refactor:` - Code restructuring
- ⚡ `Perf:` - Performance improvements
- 🧪 `Test:` - Test additions/fixes

## File Structure

```
vggt-dataset-builder/
├── *.py                    # Main scripts
├── dataset_utils.py        # Shared utilities
├── tests/                  # Test suite
├── vggt/                   # Submodule (don't modify)
├── pyproject.toml          # Project config (format, test settings)
├── .pre-commit-config.yaml # Pre-commit hooks config
├── .copilot-instructions.md # Copilot guidance
└── .vscode/                # VS Code settings
```

## Questions?

1. Check `.copilot-instructions.md` for tool usage
2. Review `README.md` for project overview
3. Look at existing code for style examples
4. Open an issue if unsure

Happy coding! 🚀
