# Contributing to DeepAR Trade Forecast

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DeepAR_trade_forecast
   ```

2. **Install dependencies**
   ```bash
   poetry install
   # or
   pip install -e ".[dev]"
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v --cov=deepar_forecast
   ```

## Code Style

- Use `black` for code formatting: `black src/ tests/`
- Use `ruff` for linting: `ruff check src/ tests/`
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write docstrings for all public functions and classes

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Test critical paths:
  - Data leakage prevention
  - Model likelihood computation
  - Backtest cost calculations

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear, atomic commits
3. Add tests for new functionality
4. Update documentation as needed
5. Run tests and linting
6. Submit PR with clear description

## Areas for Contribution

### High Priority
- Additional data providers (Alpaca, Interactive Brokers, etc.)
- More baseline models (Prophet, ARIMA improvements)
- Enhanced UI features (more charts, alerts)
- Performance optimizations (batching, caching)

### Medium Priority
- Additional trading strategies
- More evaluation metrics
- Hyperparameter tuning utilities
- Multi-symbol training improvements

### Documentation
- Tutorial notebooks
- More detailed docstrings
- Architecture diagrams
- Performance benchmarks

## Research Extensions

If you're interested in research contributions:

- Alternative likelihood functions (skew-t, GAS models)
- Attention mechanisms for DeepAR
- Multi-horizon forecasting improvements
- Portfolio optimization integration
- Risk-aware position sizing

## Questions?

Open an issue for questions or discussions about:
- Feature requests
- Bug reports
- Architecture decisions
- Research directions

## Code of Conduct

- Be respectful and constructive
- Focus on the problem, not the person
- Accept and provide constructive feedback
- Prioritize the project's goals

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
