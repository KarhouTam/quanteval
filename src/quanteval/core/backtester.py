"""
Backtester - 回测引擎
Core backtesting engine for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Union, List, cast
from quanteval.core.strategy import Strategy
from quanteval.strategies.buy_hold import BuyAndHoldStrategy
from quanteval.core.transaction import TransactionCost
from quanteval.metrics.performance import PerformanceMetrics


class MetricsSnapshot:
    """Immutable snapshot of computed performance metrics."""

    def __init__(self, metrics: Dict[str, float]):
        self._metrics = metrics

    def get_metric(self, name: str, default: float = np.nan) -> float:
        """Return a named metric value, or *default* if not present."""
        return self._metrics.get(name, default)

    def to_series(self) -> pd.Series:
        """Return all metrics as a pandas.Series."""
        return pd.Series(self._metrics)

    def to_dict(self) -> Dict[str, float]:
        """Return a shallow copy of the metrics dictionary."""
        return self._metrics.copy()

    def __getattr__(self, name: str) -> float:
        if name in self._metrics:
            return self._metrics[name]
        raise AttributeError(name)


def _metric_value(summary: pd.Series, key: str, default: float = np.nan) -> float:
    value = summary.get(key, default)
    if value is None or isinstance(value, (pd.Series, pd.DataFrame)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class BacktestResults:
    """
    回测结果类 - 存储和展示回测结果

    Backtest results container with analysis and visualization methods.
    """

    def __init__(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        positions: pd.Series,
        signals: pd.Series,
        returns: pd.Series,
        equity_curve: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_equity: Optional[pd.Series] = None,
        transaction_costs: Optional[pd.Series] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        self.strategy_name = strategy_name
        self.data = data
        self.positions = positions
        self.signals = signals
        self.returns = returns
        self.equity_curve = equity_curve
        self.benchmark_returns = benchmark_returns
        self.benchmark_equity = benchmark_equity
        self.transaction_costs = transaction_costs
        self.metrics = metrics or {}

    def summary(self) -> pd.Series:
        """
        返回绩效指标摘要

        Return performance metrics summary.
        """
        if not self.metrics:
            # Calculate metrics if not already done
            perf = PerformanceMetrics(
                strategy_returns=self.returns,
                benchmark_returns=self.benchmark_returns,
                positions=self.positions,
            )
            self.metrics = perf.calculate_all()

        return pd.Series(self.metrics)

    def get_metrics(self) -> MetricsSnapshot:
        """Return a MetricsSnapshot of all computed performance metrics."""
        return MetricsSnapshot(self.summary().to_dict())

    def plot(self, figsize=(15, 10), interactive=False):
        """
        绘制回测结果图表

        Plot backtest results (equity curve + drawdown).

        Args:
            figsize: Figure size for matplotlib
            interactive: If True, use plotly (interactive); else matplotlib (static)
        """
        if interactive:
            from quanteval.visualization.interactive import plot_backtest_results

            return plot_backtest_results(self)
        else:
            from quanteval.visualization.static import plot_backtest_results

            return plot_backtest_results(self, figsize=figsize)

    def __repr__(self) -> str:
        summary = self.summary()
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None

        if tabulate:
            information_ratio = _metric_value(summary, 'information_ratio')
            win_rate = _metric_value(summary, 'win_rate')
            total_return = _metric_value(summary, 'total_return', 0.0)
            annual_return = _metric_value(summary, 'annual_return', 0.0)
            benchmark_return = _metric_value(summary, 'benchmark_return', 0.0)
            alpha = _metric_value(summary, 'alpha', 0.0)
            annual_volatility = _metric_value(summary, 'annual_volatility', 0.0)
            max_drawdown = _metric_value(summary, 'max_drawdown', 0.0)
            max_drawdown_duration = _metric_value(summary, 'max_drawdown_duration', 0.0)
            sharpe_ratio = _metric_value(summary, 'sharpe_ratio', 0.0)
            sortino_ratio = _metric_value(summary, 'sortino_ratio', 0.0)
            calmar_ratio = _metric_value(summary, 'calmar_ratio', 0.0)
            num_trades = _metric_value(summary, 'num_trades', 0.0)
            profit_loss_ratio = _metric_value(summary, 'profit_loss_ratio', 0.0)
            avg_trade_duration = _metric_value(summary, 'avg_trade_duration', 0.0)
            table_data = [
                ['Returns', 'Total Return', f'{total_return:.2%}'],
                ['Returns', 'Annual Return', f'{annual_return:.2%}'],
                ['Returns', 'Benchmark Return', f'{benchmark_return:.2%}'],
                ['Returns', 'Alpha', f'{alpha:.2%}'],
                ['Risk', 'Annual Volatility', f'{annual_volatility:.2%}'],
                ['Risk', 'Max Drawdown', f'{max_drawdown:.2%}'],
                ['Risk', 'Max DD Duration', f'{max_drawdown_duration:.0f} days'],
                ['Risk-Adjusted', 'Sharpe Ratio', f'{sharpe_ratio:.3f}'],
                ['Risk-Adjusted', 'Sortino Ratio', f'{sortino_ratio:.3f}'],
                ['Risk-Adjusted', 'Calmar Ratio', f'{calmar_ratio:.3f}'],
            ]

            if not np.isnan(information_ratio):
                table_data.append(
                    ['Risk-Adjusted', 'Information Ratio', f'{information_ratio:.3f}']
                )

            table_data.append(['Trading', 'Number of Trades', f'{num_trades:.0f}'])

            if not np.isnan(win_rate):
                table_data.append(['Trading', 'Win Rate', f'{win_rate:.2%}'])
                table_data.append(['Trading', 'Profit/Loss Ratio', f'{profit_loss_ratio:.2f}'])
                table_data.append(
                    ['Trading', 'Avg Trade Duration', f'{avg_trade_duration:.1f} days']
                )

            table_str = tabulate(
                table_data, headers=['Category', 'Metric', 'Value'], tablefmt='pretty'
            )

            output = [
                f'\n{"=" * 60}',
                f'  {self.strategy_name} - Backtest Results',
                f'{"=" * 60}\n',
                table_str,
                f'\n{"=" * 60}\n',
            ]
            return '\n'.join(output)

        else:
            # Fallback to old formatting if tabulate is not installed
            information_ratio = _metric_value(summary, 'information_ratio')
            win_rate = _metric_value(summary, 'win_rate')
            total_return = _metric_value(summary, 'total_return', 0.0)
            annual_return = _metric_value(summary, 'annual_return', 0.0)
            benchmark_return = _metric_value(summary, 'benchmark_return', 0.0)
            alpha = _metric_value(summary, 'alpha', 0.0)
            annual_volatility = _metric_value(summary, 'annual_volatility', 0.0)
            max_drawdown = _metric_value(summary, 'max_drawdown', 0.0)
            max_drawdown_duration = _metric_value(summary, 'max_drawdown_duration', 0.0)
            sharpe_ratio = _metric_value(summary, 'sharpe_ratio', 0.0)
            sortino_ratio = _metric_value(summary, 'sortino_ratio', 0.0)
            calmar_ratio = _metric_value(summary, 'calmar_ratio', 0.0)
            num_trades = _metric_value(summary, 'num_trades', 0.0)
            profit_loss_ratio = _metric_value(summary, 'profit_loss_ratio', 0.0)
            avg_trade_duration = _metric_value(summary, 'avg_trade_duration', 0.0)
            output = [f'\n{"=" * 60}']
            output.append(f'  {self.strategy_name} - Backtest Results')
            output.append(f'{"=" * 60}\n')

            output.append('📊 Returns Metrics:')
            output.append(f'  Total Return:        {total_return:.2%}')
            output.append(f'  Annual Return:       {annual_return:.2%}')

            if 'benchmark_return' in summary:
                output.append(f'  Benchmark Return:    {benchmark_return:.2%}')
                output.append(f'  Alpha:               {alpha:.2%}')

            output.append('\n📉 Risk Metrics:')
            output.append(f'  Annual Volatility:   {annual_volatility:.2%}')
            output.append(f'  Max Drawdown:        {max_drawdown:.2%}')
            output.append(f'  Max DD Duration:     {max_drawdown_duration:.0f} days')

            output.append('\n⚖️  Risk-Adjusted Metrics:')
            output.append(f'  Sharpe Ratio:        {sharpe_ratio:.3f}')
            output.append(f'  Sortino Ratio:       {sortino_ratio:.3f}')
            output.append(f'  Calmar Ratio:        {calmar_ratio:.3f}')

            if not np.isnan(information_ratio):
                output.append(f'  Information Ratio:   {information_ratio:.3f}')

            output.append('\n📈 Trading Statistics:')
            output.append(f'  Number of Trades:    {num_trades:.0f}')

            if not np.isnan(win_rate):
                output.append(f'  Win Rate:            {win_rate:.2%}')
                output.append(f'  Profit/Loss Ratio:   {profit_loss_ratio:.2f}')
                output.append(f'  Avg Trade Duration:  {avg_trade_duration:.1f} days')

            output.append(f'\n{"=" * 60}\n')

            return '\n'.join(output)


class MultiBacktestResults:
    """
    多策略回测结果容器

    Container for multiple backtest results.
    """

    def __init__(self, results: Dict[str, BacktestResults]):
        self.results = results

    def summary(self) -> pd.DataFrame:
        """
        返回所有策略的绩效指标对比矩阵

        Return performance metrics comparison matrix for all strategies.
        """
        summaries = {}
        for name, res in self.results.items():
            summaries[name] = res.summary()
        return pd.DataFrame(summaries)

    def plot(self, figsize=(15, 10), interactive=False):
        """
        绘制所有策略的净值曲线对比

        Plot equity curves comparison for all strategies.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        benchmark_plotted = False

        for name, res in self.results.items():
            plt.plot(res.equity_curve.index, res.equity_curve, label=name)

            if not benchmark_plotted and res.benchmark_equity is not None:
                plt.plot(
                    res.equity_curve.index,
                    res.benchmark_equity,
                    label='Benchmark',
                    color='gray',
                    linestyle='--',
                    alpha=0.7,
                )
                benchmark_plotted = True

        plt.title('Multi-Strategy Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Equity (Normalized to 1.0)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        if not self.results:
            return 'MultiBacktestResults(empty)'

        summary_df = self.summary().T

        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None

        cols_map = {
            'total_return': ('Total Return', '.2%'),
            'annual_return': ('Annual Return', '.2%'),
            'benchmark_return': ('Benchmark', '.2%'),
            'alpha': ('Alpha', '.2%'),
            'annual_volatility': ('Annual Vol', '.2%'),
            'max_drawdown': ('Max Drawdown', '.2%'),
            'sharpe_ratio': ('Sharpe', '.3f'),
            'win_rate': ('Win Rate', '.2%'),
        }

        available_cols = [c for c in cols_map.keys() if c in summary_df.columns]

        if available_cols:
            display_df = cast(pd.DataFrame, summary_df[available_cols].copy())
            for c in available_cols:
                fmt = cols_map[c][1]
                column = cast(pd.Series, display_df[c])
                display_df.loc[:, c] = column.map(
                    lambda x: format(x, fmt) if pd.notnull(x) else 'NaN'
                )

            renamed_columns = {c: cols_map[c][0] for c in available_cols}
            display_df.columns = [
                renamed_columns.get(str(column), str(column)) for column in display_df.columns
            ]

            if tabulate:
                table_str = tabulate(display_df, headers='keys', tablefmt='pretty', showindex=True)
                output = [
                    f'\n{"=" * 90}',
                    f'  Multi-Strategy Backtest Results (Strategies: {len(self.results)})',
                    f'{"=" * 90}\n',
                    table_str,
                    f'\n{"=" * 90}\n',
                ]
                return '\n'.join(output)
            else:
                output = [
                    f'MultiBacktestResults (Strategies: {len(self.results)})',
                    '=' * 80,
                    str(display_df),
                    '=' * 80,
                ]
                return '\n'.join(output)

        return f'MultiBacktestResults(strategies={list(self.results.keys())})\n\n{str(summary_df)}'


class Backtester:
    """
    回测引擎 - 执行策略回测

    Backtesting engine for strategy evaluation.

    Args:
        strategy: Strategy instance or list/dict of strategies
        data: DataFrame with OHLCV data
        benchmark_strategy: Optional Strategy for benchmark (defaults to BuyAndHoldStrategy)
        transaction_costs: TransactionCost instance or bool
        initial_capital: Initial capital in CNY
    """

    def __init__(
        self,
        strategy: Union[Strategy, List[Strategy], Dict[str, Strategy]],
        data: pd.DataFrame,
        benchmark_strategy: Optional[Strategy] = None,
        transaction_costs: Union[bool, TransactionCost] = True,
        initial_capital: float = 100000.0,
        **kwargs,
    ):
        legacy_benchmark = kwargs.get('benchmark')
        if legacy_benchmark is None and 'benchmark_data' in kwargs:
            legacy_benchmark = kwargs['benchmark_data']

        legacy_transaction_cost = kwargs.get('transaction_cost')
        if legacy_transaction_cost is not None and transaction_costs is True:
            transaction_costs = legacy_transaction_cost

        # Support old `strategy` kwarg
        if 'strategy' in kwargs and not isinstance(strategy, (Strategy, list, dict)):
            strategy = kwargs['strategy']

        # Support old `benchmark` kwarg
        self.benchmark = legacy_benchmark

        if isinstance(strategy, Strategy):
            self.strategies = {strategy.name: strategy}
            self._single = True
        elif isinstance(strategy, list):
            self.strategies = {s.name: s for s in strategy}
            self._single = False
        elif isinstance(strategy, dict):
            self.strategies = strategy
            self._single = False
        else:
            raise ValueError(
                'strategies must be a Strategy, List[Strategy], or Dict[str, Strategy]'
            )

        self.data = self._prepare_data(data, label='data')
        self.benchmark_returns: Optional[pd.Series] = None
        self.benchmark_equity: Optional[pd.Series] = None

        if benchmark_strategy is None and self.benchmark is None:
            self.benchmark_strategy = BuyAndHoldStrategy()
        else:
            self.benchmark_strategy = benchmark_strategy

        self.initial_capital = initial_capital

        # Initialize transaction cost model
        if isinstance(transaction_costs, TransactionCost):
            self.transaction_cost_model = transaction_costs
        elif transaction_costs is True:
            # Use default A-share costs
            self.transaction_cost_model = TransactionCost()
        else:
            self.transaction_cost_model = None

    @staticmethod
    def _prepare_data(data: pd.DataFrame, label: str) -> pd.DataFrame:
        if 'Close' not in data.columns:
            raise ValueError(f"{label} must contain a 'Close' column")

        prepared = data.copy()
        close_values = prepared['Close']
        if isinstance(close_values, pd.DataFrame):
            raise ValueError(f"{label} contains duplicate 'Close' columns")

        if 'Ret' not in prepared.columns:
            prepared['Ret'] = cast(pd.Series, close_values).pct_change().fillna(0.0)
        else:
            ret_values = prepared['Ret']
            if isinstance(ret_values, pd.DataFrame):
                raise ValueError(f"{label} contains duplicate 'Ret' columns")
            prepared['Ret'] = cast(pd.Series, ret_values).fillna(0.0)

        return prepared

    def _run_single(self, strategy: Strategy) -> BacktestResults:
        """Run backtest for a single strategy"""
        signals = strategy.generate_signals(self.data)

        if not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=self.data.index)

        positions = cast(pd.Series, signals.shift(1).fillna(0.0))
        ret_series = cast(pd.Series, self.data['Ret'])
        close_series = cast(pd.Series, self.data['Close'])
        strategy_returns_gross = positions * ret_series

        if self.transaction_cost_model is not None:
            costs = self.transaction_cost_model.calculate(
                positions=positions, prices=close_series, initial_capital=self.initial_capital
            )
            strategy_returns = strategy_returns_gross - costs
        else:
            strategy_returns = strategy_returns_gross
            costs = pd.Series(0.0, index=self.data.index)

        strategy_returns_series = pd.Series(strategy_returns, index=self.data.index, dtype=float)
        equity_curve = cast(pd.Series, (strategy_returns_series + 1.0).cumprod())

        perf = PerformanceMetrics(
            strategy_returns=strategy_returns_series,
            benchmark_returns=self.benchmark_returns,
            positions=positions,
        )

        metrics = perf.calculate_all()

        return BacktestResults(
            strategy_name=strategy.name,
            data=self.data,
            positions=positions,
            signals=signals,
            returns=strategy_returns_series,
            equity_curve=equity_curve,
            benchmark_returns=self.benchmark_returns,
            benchmark_equity=self.benchmark_equity,
            transaction_costs=costs,
            metrics=metrics,
        )

    def run(self) -> Union[BacktestResults, MultiBacktestResults]:
        """
        运行回测

        Run backtest and return results.

        Returns:
            BacktestResults instance (if single strategy) or MultiBacktestResults
        """
        # Pre-compute benchmark
        if self.benchmark is not None:
            benchmark_df = self._prepare_data(self.benchmark, label='benchmark')
            benchmark_ret = cast(pd.Series, benchmark_df['Ret'])
            self.benchmark_returns = benchmark_ret.reindex(self.data.index).fillna(0)
            benchmark_returns = cast(pd.Series, self.benchmark_returns)
            self.benchmark_equity = cast(pd.Series, (1.0 + benchmark_returns).cumprod())
        elif self.benchmark_strategy is not None:
            bench_signals = self.benchmark_strategy.generate_signals(self.data)
            if not isinstance(bench_signals, pd.Series):
                bench_signals = pd.Series(bench_signals, index=self.data.index)
            bench_positions = cast(pd.Series, bench_signals.shift(1).fillna(0.0))
            ret_series = cast(pd.Series, self.data['Ret'])
            close_series = cast(pd.Series, self.data['Close'])
            bench_returns_gross = bench_positions * ret_series

            if self.transaction_cost_model is not None:
                bench_costs = self.transaction_cost_model.calculate(
                    positions=bench_positions,
                    prices=close_series,
                    initial_capital=self.initial_capital,
                )
                self.benchmark_returns = bench_returns_gross - bench_costs
            else:
                self.benchmark_returns = bench_returns_gross

            benchmark_returns = cast(pd.Series, self.benchmark_returns)
            self.benchmark_equity = (1.0 + benchmark_returns).cumprod()
        else:
            self.benchmark_returns = None
            self.benchmark_equity = None

        results = {}
        for name, strategy in self.strategies.items():
            results[name] = self._run_single(strategy)

        if self._single:
            return list(results.values())[0]
        else:
            return MultiBacktestResults(results)
