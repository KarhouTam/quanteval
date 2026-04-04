# QuantEval 使用教程

本教程面向第一次接触 QuantEval 的用户，重点说明如何安装、运行、扩展与验证项目。

## 1. 安装

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install -e ".[dev]"
```

## 2. 典型工作流

### 2.1 获取数据

```python
from quanteval import DataLoader

loader = DataLoader(cache_dir='./data_cache', use_cache=True)
data = loader.load_stock('600519', '20200101', '20231231')
```

### 2.2 运行回测

```python
from quanteval import Backtester
from quanteval.strategies import DualMAStrategy

strategy = DualMAStrategy(fast_window=10, slow_window=60)
results = Backtester(strategy=strategy, data=data, transaction_costs=True).run()
print(results.summary())
```

### 2.3 策略对比

```python
from quanteval import Backtester
from quanteval.strategies import BollingerMeanReversionStrategy, DualMAStrategy

comparison = Backtester(
    strategy={
        'DualMA(5,20)': DualMAStrategy(5, 20),
        'BollingerMR': BollingerMeanReversionStrategy(window=20, num_std=2.0),
    },
    data=data,
    transaction_costs=False,
).run()

print(comparison.metrics_df)
```

### 2.4 参数优化

```python
from quanteval.optimization import GridSearch
from quanteval.strategies import DualMAStrategy

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

### 2.5 Walk-forward 样本外验证

```python
from quanteval.optimization import WalkForwardAnalysis
from quanteval.strategies import DualMAStrategy

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

## 3. 推荐阅读顺序

1. `README_CN.md`：快速了解项目定位
2. `examples/01_quickstart.ipynb`：首次跑通回测
3. `examples/02_parameter_optimization.ipynb`：学习参数搜索与样本外测试
4. `examples/03_strategy_comparison.ipynb`：学习策略横向比较
5. `examples/04_hk_market_support.ipynb`：体验港股数据加载与特色交易费率

## 4. 常见建议

- 自定义策略时只生成当日信号，不要手动 `shift(1)`。
- 如果输入数据没有 `Ret` 列，`Backtester` 会自动用 `Close.pct_change()` 计算。
- 如果要作为开源项目发布，优先把通用逻辑写进 `src/quanteval`，不要只留在 Notebook 里。

## 5. 发布前检查

```bash
ruff check .
pytest
python -m build
twine check dist/*
```
