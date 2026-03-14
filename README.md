# QuantEval

![CI](https://github.com/KarhouTam/quanteval/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

[中文说明 / Chinese Guide](./docs/README_CN.md) · [贡献指南 / Contributing](./CONTRIBUTING.md)

QuantEval is a lightweight Python toolkit for researching Chinese A-share strategies, factors, and benchmark comparisons. It combines a small reusable package under `src/quanteval`, tutorial notebooks, and GitHub automation so the project is easier to publish, test, and maintain.

## Highlights

- Backtesting engine with delayed signal execution to reduce look-ahead bias
- Built-in strategies: dual moving average, Bollinger mean reversion, buy-and-hold
- Technical factors including SMA, EMA, RSI, MACD, ATR, momentum, ROC, and more
- Strategy comparison, grid-search optimization, and walk-forward analysis
- Static and interactive plots for equity curves and drawdowns
- Caching support for AkShare stock, ETF, and index data

## Repository layout

```text
.
├── src/quanteval/            # Python package
├── tests/                    # Unit tests
├── examples/                 # Tutorial notebooks
├── .github/workflows/        # CI / release automation
├── README.md                 # English overview
├── README_CN.md              # Chinese overview
└── CONTRIBUTING.md           # Developer workflow
```

## Installation

### End users

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install .
```

### Contributors

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install -e ".[dev]"
```

## Quick start

```python
from quanteval import Backtester, DataLoader, DualMAStrategy

loader = DataLoader()
data = loader.load_stock('600519', '20200101', '20231231')

strategy = DualMAStrategy(fast_window=10, slow_window=60)
results = Backtester(strategy=strategy, data=data, transaction_costs=True).run()

print(results.summary())
results.plot()
```

## More examples

- `examples/01_quickstart.ipynb`: basic data loading and first backtest
- `examples/02_parameter_optimization.ipynb`: grid search and walk-forward workflow
- `examples/03_strategy_comparison.ipynb`: compare strategies and build an equal-weight portfolio
- [TUTORIAL_EN.md](./docs/TUTORIAL_EN.md): step-by-step written tutorial in English

## Quality checks

Run the same checks used in CI before publishing:

```bash
ruff check .
pytest
python -m build
twine check dist/*
```

## GitHub automation

This repository now includes workflows for:

- linting and static checks
- unit tests across supported Python versions
- notebook execution smoke tests
- source distribution / wheel build validation
- GitHub release creation when a version tag is pushed

## License

MIT
