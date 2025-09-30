# Contributing Guide

Thank you for considering contributing to the Market Regime Detection & Adaptive Strategy Selection project!

## Getting Started

### 1. Fork and Clone
```bash
git clone https://github.com/Sakeeb91/regime-detection-strategy.git
cd regime-detection-strategy
```

### 2. Set Up Development Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write descriptive docstrings (Google style)

### Running Code Quality Checks
```bash
# Format code
black src/ tests/

# Lint code
pylint src/
flake8 src/

# Type checking
mypy src/
```

### Writing Tests
- Write unit tests for all new functionality
- Maintain >80% test coverage
- Place tests in `tests/unit/` or `tests/integration/`

```bash
# Run tests
pytest

# Check coverage
pytest --cov=src --cov-report=html
```

### Commit Guidelines
Follow conventional commit format:
```
feat: add new regime detection algorithm
fix: resolve issue with feature scaling
docs: update API documentation
test: add tests for GMM detector
refactor: simplify data preprocessing logic
```

## Pull Request Process

### 1. Ensure Tests Pass
```bash
pytest --cov=src
```

### 2. Update Documentation
- Update docstrings
- Add examples if introducing new features
- Update relevant markdown files in `docs/`

### 3. Submit Pull Request
- Provide clear description of changes
- Reference any related issues
- Include screenshots for UI changes

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe tests performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
```

## Code Review Process

- All PRs require review before merging
- Address reviewer feedback promptly
- Keep PRs focused and reasonably sized

## Reporting Issues

### Bug Reports
Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)

### Feature Requests
Include:
- Use case description
- Proposed solution
- Potential alternatives

## Questions?

Open an issue with the `question` label or reach out to the maintainers.

---

*Last Updated: 2025-09-30*