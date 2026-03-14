"""Visualization module initialization."""

from quanteval.visualization.static import (
    plot_backtest_results as plot_backtest_static,
    plot_equity_curve,
    setup_chinese_plot_style,
)

from quanteval.visualization.interactive import plot_backtest_results as plot_backtest_interactive

__all__ = [
    'plot_backtest_static',
    'plot_backtest_interactive',
    'plot_equity_curve',
    'setup_chinese_plot_style',
]
