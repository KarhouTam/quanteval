"""
Multi-Strategy Comparison Framework

This module provides tools for comparing multiple trading strategies side-by-side,
analyzing their correlations, and constructing simple portfolio combinations.
"""

from typing import List, Dict, Optional
from typing import cast
from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

from ..core.strategy import Strategy
from ..core.backtester import Backtester, BacktestResults
from ..core.transaction import TransactionCost
from .helpers import configure_chinese_font


@dataclass
class StrategyComparison:
    """Results from comparing multiple strategies."""

    strategy_names: List[str]
    results: Dict[str, BacktestResults]
    metrics_df: pd.DataFrame
    returns_df: pd.DataFrame
    equity_df: pd.DataFrame
    correlation_matrix: pd.DataFrame

    def summary(self) -> str:
        """Return formatted summary of strategy comparison."""
        lines = [
            '=' * 80,
            '  Multi-Strategy Comparison',
            '=' * 80,
            '',
            f'Strategies Compared: {len(self.strategy_names)}',
            '',
            '📊 Performance Summary:',
            '',
        ]

        # Key metrics table
        key_metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        display_df = self.metrics_df[key_metrics].copy()

        # Format percentages
        for col in ['total_return', 'annual_return', 'max_drawdown', 'win_rate']:
            if col in display_df.columns:
                display_df[col] = display_df[col] * 100

        lines.append(display_df.to_string())
        lines.append('')

        # Find best strategy for each metric
        lines.append('🏆 Best Performers:')
        lines.append('')

        best_return = self.metrics_df['total_return'].idxmax()
        best_sharpe = self.metrics_df['sharpe_ratio'].idxmax()
        best_drawdown = self.metrics_df['max_drawdown'].idxmax()  # Closest to 0

        lines.extend(
            [
                f'  Highest Return:      {best_return} ({self.metrics_df.loc[best_return, "total_return"] * 100:.2f}%)',
                f'  Best Sharpe Ratio:   {best_sharpe} ({self.metrics_df.loc[best_sharpe, "sharpe_ratio"]:.3f})',
                f'  Smallest Drawdown:   {best_drawdown} ({self.metrics_df.loc[best_drawdown, "max_drawdown"] * 100:.2f}%)',
                '',
            ]
        )

        # Correlation insights
        lines.append('🔗 Strategy Correlation:')
        lines.append('')

        # Find least correlated pair
        corr_values = []
        for i in range(len(self.strategy_names)):
            for j in range(i + 1, len(self.strategy_names)):
                corr_values.append(
                    (
                        self.strategy_names[i],
                        self.strategy_names[j],
                        self.correlation_matrix.iloc[i, j],
                    )
                )

        if corr_values:
            least_corr = min(corr_values, key=lambda x: x[2])
            most_corr = max(corr_values, key=lambda x: x[2])

            lines.extend(
                [
                    f'  Least Correlated:  {least_corr[0]} ↔ {least_corr[1]} ({least_corr[2]:.3f})',
                    f'  Most Correlated:   {most_corr[0]} ↔ {most_corr[1]} ({most_corr[2]:.3f})',
                    '',
                ]
            )

        return '\n'.join(lines)

    def get_rankings(self, metric: str = 'sharpe_ratio', ascending: bool = False) -> pd.Series:
        """
        Rank strategies by a specific metric.

        Parameters
        ----------
        metric : str
            Metric to rank by
        ascending : bool
            If True, rank from lowest to highest

        Returns
        -------
        pd.Series
            Ranked metric values
        """
        metric_series = cast(pd.Series, self.metrics_df[metric])
        return metric_series.sort_values(axis=0, ascending=ascending)

    def plot_equity_curves(self, figsize: tuple = (18, 8), style: str = 'ggplot') -> Figure:
        """
        Plot equity curves for all strategies.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        style : str
            Matplotlib style

        Returns
        -------
        Figure
            Matplotlib figure
        """
        configure_chinese_font()
        plt.style.use(style)

        fig, ax = plt.subplots(figsize=figsize)

        for strategy_name in self.strategy_names:
            equity = cast(pd.Series, self.equity_df[strategy_name])
            ax.plot(
                list(equity.index), equity.tolist(), label=strategy_name, linewidth=2, alpha=0.8
            )

        ax.set_xlabel('日期', fontsize=16)
        ax.set_ylabel('收益曲线', fontsize=16)
        ax.set_title('策略收益曲线对比', fontsize=18, fontweight='bold')
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, figsize: tuple = (10, 8)) -> Figure:
        """
        Plot correlation heatmap of strategy returns.

        Parameters
        ----------
        figsize : tuple
            Figure size

        Returns
        -------
        Figure
            Matplotlib figure
        """
        configure_chinese_font()

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(self.correlation_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(self.strategy_names)))
        ax.set_yticks(np.arange(len(self.strategy_names)))
        ax.set_xticklabels(self.strategy_names)
        ax.set_yticklabels(self.strategy_names)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Add correlation values
        for i in range(len(self.strategy_names)):
            for j in range(len(self.strategy_names)):
                ax.text(
                    j,
                    i,
                    f'{self.correlation_matrix.iloc[i, j]:.2f}',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=14,
                )

        ax.set_title('策略收益相关性矩阵', fontsize=18, fontweight='bold', pad=20)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('相关系数', rotation=270, labelpad=20)

        plt.tight_layout()
        return fig

    def plot_metrics_comparison(
        self, metrics: Optional[List[str]] = None, figsize: tuple = (18, 10)
    ) -> Figure:
        """
        Plot bar chart comparing multiple metrics across strategies.

        Parameters
        ----------
        metrics : List[str], optional
            Metrics to compare. Defaults to key metrics.
        figsize : tuple
            Figure size

        Returns
        -------
        Figure
            Matplotlib figure
        """
        if metrics is None:
            metrics = [
                'total_return',
                'annual_return',
                'sharpe_ratio',
                'sortino_ratio',
                'max_drawdown',
                'win_rate',
            ]

        configure_chinese_font()
        plt.style.use('ggplot')

        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]

        metric_labels = {
            'total_return': '总收益率 (%)',
            'annual_return': '年化收益率 (%)',
            'sharpe_ratio': '夏普比率',
            'sortino_ratio': '索提诺比率',
            'calmar_ratio': '卡玛比率',
            'max_drawdown': '最大回撤 (%)',
            'win_rate': '胜率 (%)',
            'profit_loss_ratio': '盈亏比',
        }

        for idx, metric in enumerate(metrics):
            if metric not in self.metrics_df.columns:
                continue

            ax = axes[idx]
            values = self.metrics_df[metric].copy()

            # Convert to percentage for certain metrics
            if metric in ['total_return', 'annual_return', 'max_drawdown', 'win_rate']:
                values = values * 100

            bars = ax.bar(self.strategy_names, values, alpha=0.7)

            # Color bars based on value (green for positive, red for negative)
            for i, (bar, val) in enumerate(zip(bars, values)):
                if metric == 'max_drawdown':
                    color = 'red' if val < 0 else 'green'
                else:
                    color = 'green' if val > 0 else 'red'
                bar.set_color(color)
                bar.set_alpha(0.7)

            ax.set_ylabel(metric_labels.get(metric, metric))
            ax.set_title(metric_labels.get(metric, metric), fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f'{val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=13,
                )

        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('策略指标对比', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig


class StrategyComparator:
    """
    Multi-Strategy Comparison Tool

    Compare multiple trading strategies on the same dataset and analyze
    their relative performance, correlations, and potential for combination.

    Parameters
    ----------
    strategies : List[Strategy]
        List of strategy instances to compare
    strategy_names : List[str], optional
        Custom names for strategies. If None, uses strategy class names.
    transaction_cost : TransactionCost, optional
        Transaction cost model (applied to all strategies)
    verbose : bool, default=True
        Print progress information

    Examples
    --------
    >>> from quanteval import StrategyComparator, DualMAStrategy, BollingerMeanReversionStrategy
    >>> from quanteval.data import DataLoader
    >>>
    >>> # Load data
    >>> loader = DataLoader()
    >>> data = loader.load_stock('600519', '20200101', '20231631')
    >>>
    >>> # Create strategies
    >>> strategies = [
    >>>     DualMAStrategy(fast_window=10, slow_window=60),
    >>>     DualMAStrategy(fast_window=20, slow_window=80),
    >>>     BollingerMeanReversionStrategy(window=20, num_std=2)
    >>> ]
    >>>
    >>> # Compare
    >>> comparator = StrategyComparator(strategies)
    >>> result = comparator.compare(data)
    >>>
    >>> # View results
    >>> print(result.summary())
    >>> result.plot_equity_curves()
    >>> result.plot_correlation_heatmap()
    """

    def __init__(
        self,
        strategies: List[Strategy],
        strategy_names: Optional[List[str]] = None,
        transaction_cost: Optional[TransactionCost] = None,
        verbose: bool = True,
    ):
        self.strategies = strategies
        self.transaction_cost = transaction_cost or TransactionCost()
        self.verbose = verbose

        # Generate strategy names
        if strategy_names is None:
            self.strategy_names = []
            for i, strategy in enumerate(strategies):
                name = str(strategy)
                # Add index if duplicate names
                if name in self.strategy_names:
                    name = f'{name}_{i + 1}'
                self.strategy_names.append(name)
        else:
            if len(strategy_names) != len(strategies):
                raise ValueError('strategy_names must have same length as strategies')
            self.strategy_names = strategy_names

    def compare(
        self, data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None
    ) -> StrategyComparison:
        """
        Run backtest for all strategies and compare results.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with DatetimeIndex
        benchmark_data : pd.DataFrame, optional
            Benchmark data for comparison

        Returns
        -------
        StrategyComparison
            Comparison results with metrics and visualizations
        """
        if self.verbose:
            logger.info(f'Comparing {len(self.strategies)} strategies...')

        results = {}

        # Run backtest for each strategy
        for i, (strategy, name) in enumerate(zip(self.strategies, self.strategy_names), 1):
            if self.verbose:
                logger.info(f'[{i}/{len(self.strategies)}] Running {name}...')

            backtester = Backtester(
                strategy=strategy,
                data=data,
                transaction_costs=self.transaction_cost,
                benchmark=benchmark_data,
            )

            result = backtester.run()
            if not isinstance(result, BacktestResults):
                raise RuntimeError('StrategyComparator expected single-strategy backtest results')
            results[name] = result

        # Collect metrics
        metrics_data = []
        for name, result in results.items():
            metrics = result.summary()
            metrics_dict = {
                'total_return': metrics.get('total_return', 0),
                'annual_return': metrics.get('annual_return', 0),
                'annual_volatility': metrics.get('annual_volatility', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'max_drawdown_duration': metrics.get('max_drawdown_duration', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_loss_ratio': metrics.get('profit_loss_ratio', 0),
                'num_trades': metrics.get('num_trades', 0),
            }
            metrics_data.append(metrics_dict)

        metrics_df = pd.DataFrame(metrics_data, index=self.strategy_names)

        # Collect returns and equity curves
        returns_df = pd.DataFrame({name: result.returns for name, result in results.items()})

        equity_df = pd.DataFrame({name: result.equity_curve for name, result in results.items()})

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        if self.verbose:
            logger.info('Comparison complete!')

        return StrategyComparison(
            strategy_names=self.strategy_names,
            results=results,
            metrics_df=metrics_df,
            returns_df=returns_df,
            equity_df=equity_df,
            correlation_matrix=correlation_matrix,
        )

    def create_equal_weight_portfolio(
        self, data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None
    ) -> BacktestResults:
        """
        Create an equal-weight portfolio of all strategies.

        Combines strategy returns with equal weights (1/N for N strategies).

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
        benchmark_data : pd.DataFrame, optional
            Benchmark data

        Returns
        -------
        BacktestResults
            Backtest results for the portfolio
        """
        # Run all strategies
        all_returns = []
        all_positions = []

        for strategy in self.strategies:
            backtester = Backtester(
                strategy=strategy,
                data=data,
                transaction_costs=self.transaction_cost,
                benchmark=benchmark_data,
            )
            result = backtester.run()
            if not isinstance(result, BacktestResults):
                raise RuntimeError('StrategyComparator expected single-strategy backtest results')
            all_returns.append(result.returns)
            all_positions.append(result.positions)

        # Equal weight combination
        returns_df = pd.DataFrame(all_returns).T
        portfolio_returns = returns_df.mean(axis=1)

        positions_df = pd.DataFrame(all_positions).T
        portfolio_positions = positions_df.mean(axis=1)

        # Calculate equity curve
        portfolio_equity = (1 + portfolio_returns).cumprod()

        # Get benchmark returns
        benchmark_returns = None
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.loc[
                portfolio_returns.index[0] : portfolio_returns.index[-1], 'Ret'
            ]
        benchmark_equity = None if benchmark_returns is None else (1 + benchmark_returns).cumprod()

        # Calculate metrics
        from ..metrics.performance import PerformanceMetrics

        perf = PerformanceMetrics(
            strategy_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            positions=portfolio_positions,
        )
        metrics_dict = perf.calculate_all()

        return BacktestResults(
            strategy_name='Portfolio',
            data=data.loc[portfolio_returns.index],
            positions=portfolio_positions,
            signals=pd.Series(0, index=portfolio_positions.index),
            returns=portfolio_returns,
            equity_curve=portfolio_equity,
            benchmark_returns=benchmark_returns,
            benchmark_equity=benchmark_equity,
            metrics=metrics_dict,
        )


def quick_compare(
    strategies: List[Strategy],
    data: pd.DataFrame,
    strategy_names: Optional[List[str]] = None,
    show_plots: bool = True,
) -> None:
    """
    Quick helper function for comparing strategies with automatic visualization.

    Parameters
    ----------
    strategies : List[Strategy]
        Strategies to compare
    data : pd.DataFrame
        OHLCV data
    strategy_names : List[str], optional
        Custom strategy names
    show_plots : bool
        Whether to display plots
    """
    comparator = StrategyComparator(strategies, strategy_names)
    result = comparator.compare(data)

    logger.info(result.summary())

    if show_plots:
        result.plot_equity_curves()
        result.plot_metrics_comparison()
        result.plot_correlation_heatmap()
        plt.show()
