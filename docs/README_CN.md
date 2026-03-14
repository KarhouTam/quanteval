# 📈 QuantEval 中文说明

![CI](https://github.com/KarhouTam/quanteval/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

[English README](../README.md) · [贡献指南](../CONTRIBUTING.md)

QuantEval 是一个专门为 A 股研究量身定制的轻量级、友好型 Python 工具包！无论是策略回测、因子验证，还是参数优化，它都能帮你更高效地完成，让你的量化之旅更加顺滑。🚀

## ✨ 功能概览

- ⏱️ **专业回测引擎**：支持信号延迟执行，帮你避开恼人的“未来函数”陷阱，让回测更贴近实战！
- 🤖 **内置经典策略**：双均线 (MA)、布林带 (Bollinger Mean Reversion)、以及稳定的买入持有 (Buy & Hold)。
- 📊 **丰富技术指标**：从 SMA、EMA 到 RSI、MACD、ATR 以及动量指标等一应俱全。
- 🔍 **优化与分析**：支持网格搜索优化参数，还有 walk-forward 样本外分析让你的策略更经得起考验。
- 🎨 **可视化大师**：内置漂亮且直观的静态与交互式图表，轻松查看收益曲线与回撤。
- 💾 **数据一手掌握**：无缝对接 AkShare 接口，支持股票、ETF、指数数据的本地缓存。

## 📂 仓库结构

```text
.
├── src/quanteval/            # 🧠 核心源码包
├── tests/                    # ✅ 单元测试
├── examples/                 # 📚 教学示例 Notebook
├── .github/workflows/        # 🤖 自动化 CI / 发布流程
├── README.md                 # 🌐 英文主页
├── README_CN.md              # 🏮 中文说明 (本文件)
└── CONTRIBUTING.md           # 🤝 开发者指南
```

## 🛠️ 安装方式

### 📦 来自 PyPI

最省心的方式，直接一键安装：

```bash
pip install quanteval
```

### 🔨 普通用户（从源码安装）

如果你想运行最新的 master 分支代码：

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install .
```

### 🧑‍💻 开发者安装

如果你打算为项目贡献代码：

```bash
git clone https://github.com/KarhouTam/quanteval.git
cd quanteval
python -m pip install -e ".[dev]"
```

## 🚀 快速开始

运行你的第一个 A 股回测只需要几行代码：

```python
from quanteval import Backtester, DataLoader, DualMAStrategy

# 1. 搬运点数据 🏪
loader = DataLoader()
data = loader.load_stock('600519', '20200101', '20231231')

# 2. 选个策略启动！ 🧠
strategy = DualMAStrategy(fast_window=10, slow_window=60)

# 3. 看看收益如何？ 📊
results = Backtester(strategy=strategy, data=data, transaction_costs=True).run()

print(results.summary()) # 打印关键指标
results.plot()           # 快乐看图
```

## 📖 更多干货

查看这些 Notebooks 快速上手：

- [`examples/01_quickstart.ipynb`](../examples/01_quickstart.ipynb)：新手村第一站，基础数据加载与回测 🏃
- [`examples/02_parameter_optimization.ipynb`](../examples/02_parameter_optimization.ipynb)：寻找策略“财富密码”，网格搜索与 walk-forward 🎯
- [`examples/03_strategy_comparison.ipynb`](../examples/03_strategy_comparison.ipynb)：策略大比拼，构建等权组合 ⚔️
- [`examples/04_hk_market_support.ipynb`](../examples/04_hk_market_support.ipynb)：进军港股市场，双边印花税与恒指数据 🇭🇰
- [`TUTORIAL_CN.md`](./TUTORIAL_CN.md): 深度中文图文教程 📚
