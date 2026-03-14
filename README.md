# 📈 QuantEval

![CI](https://github.com/KarhouTam/quanteval/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

[中文说明 / Chinese Guide](./docs/README_CN.md) · [贡献指南 / Contributing](./CONTRIBUTING.md)

QuantEval is a lightweight, friendly Python toolkit for researching Chinese A-share strategies, factors, and benchmark comparisons. We've combined a reusable package under `src/quanteval`, tutorial notebooks, and automated workflows to make your quant journey smoother! 🚀

## ✨ Highlights

- ⏱️ **Backtesting engine** with delayed signal execution to keep things realistic (no look-ahead bias here!).
- 🤖 **Built-in strategies**: Dual Moving Average, Bollinger Mean Reversion, and consistent Buy-and-Hold.
- 📊 **Technical factors**: From SMA and RSI to MACD, ATR, and more.
- 🔍 **Optimization**: Grid-search and walk-forward analysis to fine-tune your ideas.
- 🎨 **Visualization**: Pretty static and interactive plots for your equity curves.
- 💾 **Data Caching**: Native support for AkShare stock, ETF, and index data.

## 📂 Repository layout

```text
.
├── src/quanteval/            # 🧠 Python package core
├── tests/                    # ✅ Unit tests
├── examples/                 # 📚 Tutorial notebooks
├── .github/workflows/        # 🤖 CI / Automation
├── README.md                 # 🌐 English overview
├── README_CN.md              # 🏮 Chinese overview
└── CONTRIBUTING.md           # 🤝 Developer workflow
```

## 🛠️ Installation

### 📦 From PyPI

The package is available on PyPI. Install the latest released version with:

```bash
pip install quanteval
```

### 🔨 End users (install from source)

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install .
```

### 🧑‍💻 Contributors

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install -e ".[dev]"
```

## 🚀 Quick start

Ready to run your first backtest? It's as simple as this:

```python
from quanteval import Backtester, DataLoader, DualMAStrategy

# 1. Grab some data 🏪
loader = DataLoader()
data = loader.load_stock('600519', '20200101', '20231231')

# 2. Pick a strategy 🧠
strategy = DualMAStrategy(fast_window=10, slow_window=60)

# 3. Let's see how it performed! 📊
results = Backtester(strategy=strategy, data=data, transaction_costs=True).run()

print(results.summary())
results.plot()
```

## 📖 More examples

Check out these notebooks to level up:

- `examples/01_quickstart.ipynb`: Basic data loading and first backtest 🏃
- `examples/02_parameter_optimization.ipynb`: Finding the best parameters 🎯
- `examples/03_strategy_comparison.ipynb`: Battle of the strategies ⚔️
- [TUTORIAL_EN.md](./docs/TUTORIAL_EN.md): Deep dive tutorial 📚

## 🧪 Quality checks

Keep the code clean! Run these before pushing:

```bash
ruff check .           # ✨ Linting
pytest                 # ✅ Tests
python -m build        # 📦 Build wheel
twine check dist/*     # 🔍 Validate
```

## 🤖 GitHub Automation

We've set up some helpful bots for:

- ✅ Linting and static checks
- 🧪 Multi-version Python testing
- 📓 Notebook smoke tests
- 📦 Build validation
- 🏷️ Automatic releases on version tags
