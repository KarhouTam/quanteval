"""Visualization module - Static matplotlib charts."""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple


def setup_chinese_plot_style():
    """设置中文绘图样式 (Setup Chinese font for plots)"""
    plt.style.use('seaborn-v0_8-whitegrid')

    from quanteval.utils.helpers import configure_chinese_font

    configure_chinese_font()


def plot_backtest_results(results, figsize: Tuple[int, int] = (15, 10)):
    """
    绘制回测结果 (Plot backtest results)

    Creates a comprehensive visualization with:
        - Equity curves (strategy vs benchmark)
        - Drawdown chart
        - Position indicators

    Args:
        results: BacktestResults instance
        figsize: Figure size
    """
    setup_chinese_plot_style()

    # Create figure with 3 subpanels
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True
    )

    # Panel 1: Equity Curves
    ax1.plot(
        results.equity_curve.index,
        results.equity_curve.values,
        label=f'{results.strategy_name}',
        color='red',
        linewidth=2,
    )

    if results.benchmark_equity is not None:
        ax1.plot(
            results.benchmark_equity.index,
            results.benchmark_equity.values,
            label='基准 (Buy & Hold)',
            color='gray',
            alpha=0.6,
            linewidth=1.5,
        )

    ax1.set_ylabel('收益曲线 (Equity)', fontsize=16)
    ax1.set_title(
        f'{results.strategy_name} - 回测结果（Backtest Result）', fontsize=18, fontweight='bold'
    )
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Drawdown
    peak = results.equity_curve.cummax()
    drawdown = (results.equity_curve - peak) / peak

    ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3, label='回撤')
    ax2.plot(drawdown.index, drawdown.values, color='red', alpha=0.6, linewidth=1)
    ax2.set_ylabel('回撤 (Drawdown)', fontsize=16)
    ax2.set_ylim([drawdown.min() * 1.1, 0.05])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    # Panel 3: Position
    ax3.fill_between(
        results.positions.index,
        results.positions.values,
        0,
        where=(results.positions == 1),
        color='green',
        alpha=0.3,
        label='持仓',
    )
    ax3.set_ylabel('仓位 (Position)', fontsize=16)
    ax3.set_xlabel('日期 (Date)', fontsize=16)
    ax3.set_ylim([-0.1, 1.2])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['空仓', '满仓'])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')

    plt.tight_layout()
    return fig


def plot_equity_curve(
    strategy_equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    title: str = '策略收益曲线（Demo Strategy Equity Curve）',
    figsize: Tuple[int, int] = (16, 6),
):
    """
    绘制简单的收益曲线 (Plot simple equity curve)

    Args:
        strategy_equity: Strategy equity curve
        benchmark_equity: Benchmark equity curve (optional)
        title: Chart title
        figsize: Figure size
    """
    setup_chinese_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        list(strategy_equity.index),
        strategy_equity.tolist(),
        label='策略（Strategy）',
        color='red',
        linewidth=2,
    )

    if benchmark_equity is not None:
        ax.plot(
            list(benchmark_equity.index),
            benchmark_equity.tolist(),
            label='基准（Benchmark）',
            color='gray',
            alpha=0.6,
            linewidth=1.5,
        )

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel('日期（Date）', fontsize=16)
    ax.set_ylabel('收益（Equity）', fontsize=16)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
