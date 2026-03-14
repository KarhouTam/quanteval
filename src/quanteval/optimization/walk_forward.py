"""
Walk-Forward Analysis for Strategy Optimization

Walk-forward analysis is a robust out-of-sample testing methodology that
helps prevent overfitting by repeatedly optimizing on in-sample data and
testing on out-of-sample data in a rolling window fashion.
"""

from typing import Dict, List, Any, Optional, Tuple, cast
from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.backtester import Backtester, BacktestResults
from ..core.transaction import TransactionCost
from .grid_search import GridSearch

logger = logging.getLogger(__name__)


def _series_metric(metrics: pd.Series, name: str, default: float = np.nan) -> float:
    value = metrics.get(name, default)
    if value is None or isinstance(value, (pd.Series, pd.DataFrame)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _index_datetime(frame: pd.DataFrame, position: int) -> datetime:
    value = frame.index[position]
    if isinstance(value, pd.Timestamp):
        return cast(datetime, value.to_pydatetime())
    if isinstance(value, datetime):
        return value
    return cast(datetime, pd.Timestamp(str(value)).to_pydatetime())


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict[str, Any]
    train_score: float
    test_score: float
    test_results: BacktestResults


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""

    windows: List[WalkForwardWindow]
    combined_results: BacktestResults
    metric_name: str
    strategy_class: type

    def summary(self) -> str:
        """Return a formatted summary of walk-forward results."""
        lines = [
            '=' * 70,
            '  滚动前瞻分析结果（Walk-Forward Analysis）',
            '=' * 70,
            '',
            f'策略名称（Strategy）: {self.strategy_class.__name__}',
            f'优化指标（Optimization Metric）: {self.metric_name}',
            f'窗口数量（Number of Windows）: {len(self.windows)}',
            '',
            '📊 各窗口表现（Per-Window Performance）:',
            '',
        ]

        # Table header
        lines.append(
            f'{"窗口（Window）":<8} {"训练区间（Train Period）":<24} {"测试区间（Test Period）":<24} {"训练得分（Train）":<12} {"测试得分（Test）":<12} {"回撤（Gap）":<10}'
        )
        lines.append('-' * 86)

        for w in self.windows:
            train_period = f'{w.train_start:%Y-%m-%d} - {w.train_end:%Y-%m-%d}'
            test_period = f'{w.test_start:%Y-%m-%d} - {w.test_end:%Y-%m-%d}'
            gap = w.train_score - w.test_score

            lines.append(
                f'{w.window_id:<8} {train_period:<24} {test_period:<24} '
                f'{w.train_score:>9.3f} {w.test_score:>9.3f} {gap:>9.3f}'
            )

        # Overall statistics
        train_scores = [w.train_score for w in self.windows]
        test_scores = [w.test_score for w in self.windows]
        gaps = [w.train_score - w.test_score for w in self.windows]

        lines.extend(
            [
                '-' * 86,
                f'{"平均值（Mean）":<8} {"":<24} {"":<24} {float(np.mean(train_scores)):>9.3f} {float(np.mean(test_scores)):>9.3f} {float(np.mean(gaps)):>9.3f}',
                f'{"标准差（Std）":<8} {"":<24} {"":<24} {float(np.std(train_scores)):>9.3f} {float(np.std(test_scores)):>9.3f} {float(np.std(gaps)):>9.3f}',
                '',
                '🎯 样本外综合表现（Combined Out-of-Sample Performance）:',
                '',
            ]
        )

        # Combined metrics
        metrics = self.combined_results.summary()
        lines.extend(
            [
                f'  总收益（Total Return）:           {_series_metric(metrics, "total_return", 0.0) * 100:>8.2f}%',
                f'  年化收益（Annual Return）:         {_series_metric(metrics, "annual_return", 0.0) * 100:>8.2f}%',
                f'  夏普比率（Sharpe Ratio）:         {_series_metric(metrics, "sharpe_ratio", 0.0):>8.3f}',
                f'  最大回撤（Max Drawdown）:         {_series_metric(metrics, "max_drawdown", 0.0) * 100:>8.2f}%',
                f'  胜率（Win Rate）:                 {_series_metric(metrics, "win_rate") * 100:>8.2f}%',
                '',
            ]
        )

        return '\n'.join(lines)

    def get_parameter_stability(self) -> pd.DataFrame:
        """
        Analyze parameter stability across windows.

        Returns a DataFrame showing how often each parameter value was selected.
        """
        param_data = []
        for w in self.windows:
            param_data.append({'window_id': w.window_id, **w.best_params})

        return pd.DataFrame(param_data)

    def get_performance_consistency(self) -> Dict[str, float]:
        """
        Calculate performance consistency metrics.

        Returns
        -------
        Dict with:
        - mean_train_score: Average training score
        - mean_test_score: Average test score
        - mean_degradation: Average performance drop from train to test
        - degradation_std: Standard deviation of degradation
        - worst_degradation: Worst performance drop
        - consistency_ratio: test_mean / train_mean
        """
        train_scores = [w.train_score for w in self.windows]
        test_scores = [w.test_score for w in self.windows]
        degradations = [t - s for t, s in zip(train_scores, test_scores)]

        mean_train = float(np.mean(train_scores))
        mean_test = float(np.mean(test_scores))
        mean_degradation = float(np.mean(degradations))
        degradation_std = float(np.std(degradations))
        worst_degradation = float(np.max(degradations))
        consistency_ratio = float(mean_test / mean_train) if mean_train != 0 else 0.0
        return {
            'mean_train_score': mean_train,
            'mean_test_score': mean_test,
            'mean_degradation': mean_degradation,
            'degradation_std': degradation_std,
            'worst_degradation': worst_degradation,
            'consistency_ratio': consistency_ratio,
        }


class WalkForwardAnalysis:
    """
    Walk-Forward Analysis Optimizer

    Implements walk-forward optimization by:
    1. Splitting data into overlapping train/test windows
    2. Optimizing parameters on each training window
    3. Testing with optimal parameters on corresponding test window
    4. Combining all out-of-sample results for final evaluation

    This approach provides more realistic performance estimates and helps
    detect overfitting.

    Parameters
    ----------
    strategy_class : type
        The strategy class to optimize
    param_grid : Dict[str, List[Any]]
        Parameter grid for optimization
    train_period : int
        Number of trading days for training window
    test_period : int
        Number of trading days for testing window
    metric : str, default='sharpe_ratio'
        Metric to optimize
    anchored : bool, default=False
        If True, training window grows (anchored walk-forward)
        If False, training window slides (rolling walk-forward)
    transaction_cost : TransactionCost, optional
        Transaction cost model
    verbose : bool, default=True
        Print progress information

    Examples
    --------
    >>> from quanteval import WalkForwardAnalysis, DualMAStrategy
    >>> from quanteval.data import DataLoader
    >>>
    >>> # Load data
    >>> loader = DataLoader()
    >>> data = loader.load_stock('600519', '20180101', '20231231')
    >>>
    >>> # Define parameter grid
    >>> param_grid = {
    >>>     'fast_window': [5, 10, 15, 20],
    >>>     'slow_window': [30, 40, 50, 60]
    >>> }
    >>>
    >>> # Run walk-forward analysis
    >>> wfa = WalkForwardAnalysis(
    >>>     DualMAStrategy,
    >>>     param_grid,
    >>>     train_period=252,  # 1 year training
    >>>     test_period=63,    # 1 quarter testing
    >>>     metric='sharpe_ratio'
    >>> )
    >>> result = wfa.run(data)
    >>>
    >>> # View results
    >>> print(result.summary())
    >>> print(result.get_parameter_stability())
    """

    def __init__(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        train_period: int,
        test_period: int,
        metric: str = 'sharpe_ratio',
        anchored: bool = False,
        transaction_cost: Optional[TransactionCost] = None,
        verbose: bool = True,
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.train_period = train_period
        self.test_period = test_period
        self.metric = metric
        self.anchored = anchored
        self.transaction_cost = transaction_cost or TransactionCost()
        self.verbose = verbose

    def _split_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into train/test windows.

        Returns list of (train_data, test_data) tuples.
        """
        windows = []
        total_periods = len(data)

        if self.anchored:
            # Anchored walk-forward: training window grows
            start_idx = 0
            current_idx = self.train_period

            while current_idx + self.test_period <= total_periods:
                train_data = data.iloc[start_idx:current_idx]
                test_data = data.iloc[current_idx : current_idx + self.test_period]
                windows.append((train_data, test_data))
                current_idx += self.test_period
        else:
            # Rolling walk-forward: training window slides
            current_idx = 0

            while current_idx + self.train_period + self.test_period <= total_periods:
                train_data = data.iloc[current_idx : current_idx + self.train_period]
                test_data = data.iloc[
                    current_idx + self.train_period : current_idx
                    + self.train_period
                    + self.test_period
                ]
                windows.append((train_data, test_data))
                current_idx += self.test_period

        if self.verbose:
            logger.info(f'Created {len(windows)} walk-forward windows')
            logger.info(f'Window type: {"Anchored" if self.anchored else "Rolling"}')

        return windows

    def run(
        self, data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with DatetimeIndex
        benchmark_data : pd.DataFrame, optional
            Benchmark data for comparison

        Returns
        -------
        WalkForwardResult
            Complete walk-forward analysis results
        """
        # Split into windows
        windows_data = self._split_windows(data)

        if len(windows_data) == 0:
            raise ValueError(
                f'No windows created. Data length ({len(data)}) is too short for '
                f'train_period ({self.train_period}) + test_period ({self.test_period})'
            )

        # Process each window
        window_results = []
        all_test_results = []

        for i, (train_data, test_data) in enumerate(windows_data, 1):
            if self.verbose:
                logger.info(f'Window {i}/{len(windows_data)}')
                logger.info(
                    f'Train: {train_data.index[0]:%Y-%m-%d} to {train_data.index[-1]:%Y-%m-%d}'
                )
                logger.info(
                    f'Test:  {test_data.index[0]:%Y-%m-%d} to {test_data.index[-1]:%Y-%m-%d}'
                )

            # Optimize on training data
            grid_search = GridSearch(
                self.strategy_class,
                self.param_grid,
                metric=self.metric,
                transaction_cost=self.transaction_cost,
                verbose=self.verbose,
            )

            # Get benchmark slices if provided
            train_benchmark = None
            test_benchmark = None
            if benchmark_data is not None:
                train_benchmark = benchmark_data.loc[train_data.index[0] : train_data.index[-1]]
                test_benchmark = benchmark_data.loc[test_data.index[0] : test_data.index[-1]]

            train_result = grid_search.fit(train_data, train_benchmark)

            # Test on out-of-sample data with best parameters
            best_strategy = self.strategy_class(**train_result.best_params)
            cost_model = cast(TransactionCost, self.transaction_cost)
            backtester = Backtester(
                strategy=best_strategy,
                data=test_data,
                transaction_costs=cost_model,
                benchmark=test_benchmark,
            )
            test_results = backtester.run()
            if not isinstance(test_results, BacktestResults):
                raise RuntimeError('WalkForwardAnalysis expected single-strategy backtest results')
            test_metrics = test_results.summary()
            test_score = _series_metric(test_metrics, self.metric)

            # Handle max_drawdown (convert to positive)
            if self.metric == 'max_drawdown':
                test_score = -test_score

            if self.verbose:
                logger.info(f'Window {i} complete:')
                logger.debug(f'   Best params: {train_result.best_params}')
                logger.info(f'   Train {self.metric}: {train_result.best_score:.4f}')
                logger.info(f'   Test {self.metric}: {test_score:.4f}')
                logger.info(f'   Degradation: {train_result.best_score - test_score:.4f}')

            # Store window result
            window = WalkForwardWindow(
                window_id=i,
                train_start=_index_datetime(train_data, 0),
                train_end=_index_datetime(train_data, -1),
                test_start=_index_datetime(test_data, 0),
                test_end=_index_datetime(test_data, -1),
                best_params=train_result.best_params,
                train_score=train_result.best_score,
                test_score=test_score,
                test_results=test_results,
            )
            window_results.append(window)
            all_test_results.append(test_results)

        # Combine all out-of-sample results
        combined_data = cast(pd.DataFrame, pd.concat([r.data for r in all_test_results]))
        combined_signals = cast(pd.Series, pd.concat([r.signals for r in all_test_results]))
        combined_positions = cast(pd.Series, pd.concat([r.positions for r in all_test_results]))
        combined_returns = cast(pd.Series, pd.concat([r.returns for r in all_test_results]))
        combined_equity = cast(pd.Series, (1.0 + combined_returns).cumprod())
        combined_costs = None
        if all(r.transaction_costs is not None for r in all_test_results):
            combined_costs = cast(
                pd.Series,
                pd.concat([cast(pd.Series, r.transaction_costs) for r in all_test_results]),
            )

        # Get combined benchmark if available
        combined_benchmark_returns = None
        combined_benchmark_equity = None
        if benchmark_data is not None:
            if 'Ret' not in benchmark_data.columns:
                benchmark_data = benchmark_data.copy()
                benchmark_close = cast(pd.Series, benchmark_data['Close'])
                benchmark_data['Ret'] = benchmark_close.pct_change().fillna(0.0)
            combined_benchmark_returns = cast(
                pd.Series,
                benchmark_data.loc[combined_equity.index[0] : combined_equity.index[-1], 'Ret'],
            )
            combined_benchmark_equity = cast(
                pd.Series, (1.0 + combined_benchmark_returns).cumprod()
            )

        # Create combined results object
        from ..metrics.performance import PerformanceMetrics

        combined_metrics = PerformanceMetrics(
            strategy_returns=combined_returns,
            benchmark_returns=combined_benchmark_returns,
            positions=combined_positions,
        ).calculate_all()

        combined_results = BacktestResults(
            strategy_name=f'{self.strategy_class.__name__} WalkForward',
            data=combined_data,
            signals=combined_signals,
            equity_curve=combined_equity,
            returns=combined_returns,
            positions=combined_positions,
            benchmark_returns=combined_benchmark_returns,
            benchmark_equity=combined_benchmark_equity,
            transaction_costs=combined_costs,
            metrics=combined_metrics,
        )

        if self.verbose:
            logger.info('Walk-Forward Analysis Complete!')

        return WalkForwardResult(
            windows=window_results,
            combined_results=combined_results,
            metric_name=self.metric,
            strategy_class=self.strategy_class,
        )
