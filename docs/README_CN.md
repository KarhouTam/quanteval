# QuantEval 中文说明

![CI](https://github.com/KarhouTam/quanteval/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

[English README](../README.md) · [贡献指南](../CONTRIBUTING.md)

QuantEval 是一个面向中国 A 股研究场景的轻量级 Python 工具包，适合做策略回测、因子验证、参数优化和策略对比。仓库同时包含可复用的 `src/quanteval` 包、教学示例 Notebook，以及适合发布到 GitHub 的自动化工作流。

## 功能概览

- 支持信号延迟执行的回测引擎，降低未来函数风险
- 内置策略：双均线、布林带均值回归、买入持有
- 内置技术指标：SMA、EMA、RSI、MACD、ATR、动量、ROC 等
- 支持策略对比、网格搜索和 walk-forward 样本外分析
- 提供 matplotlib 与 plotly 两套可视化接口
- 支持 AkShare 股票、ETF、指数数据获取与本地缓存

## 仓库结构

```text
.
├── src/quanteval/            # Python 包源码
├── tests/                    # 单元测试
├── examples/                 # 教学 Notebook
├── .github/workflows/        # CI / 发布流程
├── README.md                 # 英文说明
├── README_CN.md              # 中文说明
└── CONTRIBUTING.md           # 贡献与开发流程
```

## 安装方式

### 普通用户

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install .
```

### 开发者

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install -e ".[dev]"
```

## 快速开始

```python
from quanteval import Backtester, DataLoader, DualMAStrategy

loader = DataLoader()
data = loader.load_stock('600519', '20200101', '20231231')

strategy = DualMAStrategy(fast_window=10, slow_window=60)
results = Backtester(strategy=strategy, data=data, transaction_costs=True).run()

print(results.summary())
results.plot()
```

## 示例入口

- `examples/01_quickstart.ipynb`：首次加载数据并运行回测
- `examples/02_parameter_optimization.ipynb`：参数搜索与 walk-forward 分析
- `examples/03_strategy_comparison.ipynb`：多策略对比与等权组合
- [TUTORIAL_CN.md](./TUTORIAL_CN.md): 中文版教程

## 发布前建议检查

```bash
ruff check .
pytest
python -m build
twine check dist/*
```

## 已添加的 GitHub 自动化

- 代码风格检查与静态分析
- 多 Python 版本单元测试
- Notebook 执行烟雾测试
- sdist / wheel 构建校验
- 推送版本标签后自动创建 GitHub Release

## 发布到 GitHub 前请注意

- 发布 tag 前建议重新执行 Notebook，避免输出过期。
- `data_cache/` 下的大文件默认不纳入版本控制，除非你确实希望附带示例数据。

## 许可证

MIT
