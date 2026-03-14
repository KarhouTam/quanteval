# QuantEval Tutorial

This tutorial is for users new to QuantEval, focusing on how to install, run, extend, and verify the project.

## 1. Installation

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install -e ".[dev]"
```

## 2. Typical Workflow

### 2.1 Data Acquisition

```python
from quanteval import DataLoader

loader = DataLoader(cache_dir='./data_cache', use_cache=True)
data = loader.load_stock('600519', '20200101', '20231231')
```

### 2.2 Running Backtests

```python
from quanteval import Backtester, DualMAStrategy

strategy = DualMAStrategy(fast_window=10, slow_window=60)
results = Backtester(strategy=strategy, data=data, transaction_costs=True).run()
print(results.summary())
```

### 2.3 Strategy Comparison

```python
from quanteval import BollingerMeanReversionStrategy, StrategyComparator

comparison = StrategyComparator(
    strategies=[
        DualMAStrategy(5, 20),
        BollingerMeanReversionStrategy(window=20, num_std=2.0),
    ],
    verbose=False,
).compare(data)

print(comparison.metrics_df)
```

### 2.4 Parameter Optimization

```python
from quanteval import GridSearch

search = GridSearch(
    DualMAStrategy,
    {
        'fast_window': [5, 10],
        'slow_window': [20, 30],
        'ma_type': ['sma'],
    },
    verbose=False,
)

result = search.fit(data)
print(result.best_params)
```

### 2.5 Walk-forward Out-of-sample Validation

```python
from quanteval import WalkForwardAnalysis

wfa = WalkForwardAnalysis(
    DualMAStrategy,
    {
        'fast_window': [5, 10],
        'slow_window': [20, 30],
        'ma_type': ['sma'],
    },
    train_period=120,
    test_period=40,
    verbose=False,
)

wfa_result = wfa.run(data)
print(wfa_result.summary())
```

## 3. Recommended Reading Order

1. `README.md`: Quickly understand the project positioning.
2. `examples/01_quickstart.ipynb`: First successful backtest execution.
3. `examples/02_parameter_optimization.ipynb`: Learn parameter search and out-of-sample testing.
4. `examples/03_strategy_comparison.ipynb`: Learn horizontal strategy comparison.
5. `examples/04_hk_market_support.ipynb`: Using HK market data and transaction rules.

## 4. Common Tips

- When customizing strategies, only generate signals for the current day; do not manually `shift(1)`.
- If the input data lacks a `Ret` column, `Backtester` will automatically calculate it using `Close.pct_change()`.
- If publishing as an open-source project, prioritize moving general logic into `src/quanteval` rather than leaving it only in Notebooks.

## 5. Pre-publish Checklist

```bash
ruff check .
pytest
python -m build
twine check dist/*
```
