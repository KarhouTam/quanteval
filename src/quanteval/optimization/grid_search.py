"""
Grid Search Optimization for Strategy Parameters

This module provides grid search functionality to find optimal strategy parameters
by exhaustively testing all combinations of specified parameter values.
"""

from typing import Dict, List, Any, Optional, Tuple
import itertools
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import warnings

from ..core.backtester import Backtester
from ..core.transaction import TransactionCost

logger = logging.getLogger(__name__)


@dataclass
class GridSearchResult:
    """Results from a grid search optimization."""

    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    metric_name: str

    def summary(self) -> str:
        """Return a formatted summary of the grid search results."""
        lines = [
            '=' * 60,
            '  Grid Search Optimization Results',
            '=' * 60,
            '',
            '🎯 Best Parameters:',
        ]

        for param, value in self.best_params.items():
            lines.append(f'  {param}: {value}')

        lines.extend(
            [
                '',
                f'📊 Best {self.metric_name}: {self.best_score:.4f}',
                '',
                f'Total Combinations Tested: {len(self.all_results)}',
                '',
            ]
        )

        return '\n'.join(lines)

    def get_top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top N parameter combinations by score."""
        return self.all_results.nlargest(n, self.metric_name)


class GridSearch:
    """
    Grid Search Optimizer for Strategy Parameters

    Performs exhaustive search over specified parameter values to find
    the optimal combination based on a chosen performance metric.

    Parameters
    ----------
    strategy_class : type
        The strategy class to optimize (not an instance)
    param_grid : Dict[str, List[Any]]
        Dictionary mapping parameter names to lists of values to test
        Example: {'fast_window': [5, 10, 15], 'slow_window': [20, 30, 40]}
    metric : str, default='sharpe_ratio'
        The performance metric to optimize. Options include:
        - 'sharpe_ratio' (default)
        - 'sortino_ratio'
        - 'calmar_ratio'
        - 'total_return'
        - 'annual_return'
        - 'max_drawdown' (will maximize, i.e., minimize drawdown)
        - 'win_rate'
        - 'profit_loss_ratio'
    transaction_cost : TransactionCost, optional
        Transaction cost model to use
    n_jobs : int, default=1
        Number of parallel jobs. -1 means using all processors.
        Note: Parallel execution may not work well with all strategy types.
    verbose : bool, default=True
        Whether to print progress information

    Examples
    --------
    >>> from quanteval import GridSearch, DualMAStrategy
    >>> from quanteval.data import DataLoader
    >>>
    >>> # Load data
    >>> loader = DataLoader()
    >>> data = loader.load_stock('600519', '20200101', '20231231')
    >>>
    >>> # Define parameter grid
    >>> param_grid = {
    >>>     'fast_window': [5, 10, 15, 20],
    >>>     'slow_window': [30, 40, 50, 60],
    >>>     'ma_type': ['sma', 'ema']
    >>> }
    >>>
    >>> # Run grid search
    >>> grid_search = GridSearch(
    >>>     DualMAStrategy,
    >>>     param_grid,
    >>>     metric='sharpe_ratio'
    >>> )
    >>> result = grid_search.fit(data)
    >>>
    >>> # View results
    >>> print(result.summary())
    >>> print(result.get_top_n(5))
    """

    def __init__(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio',
        transaction_cost: Optional[TransactionCost] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.metric = metric
        self.transaction_cost = transaction_cost or TransactionCost()
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Validate metric
        valid_metrics = [
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
            'total_return',
            'annual_return',
            'max_drawdown',
            'win_rate',
            'profit_loss_ratio',
            'information_ratio',
        ]
        if metric not in valid_metrics:
            raise ValueError(f'metric must be one of {valid_metrics}')

        # Generate all parameter combinations
        self.param_combinations = self._generate_combinations()

        if self.verbose:
            logger.info(
                f'Grid Search initialized with {len(self.param_combinations)} parameter combinations'
            )

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _evaluate_params(
        self,
        params: Dict[str, Any],
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Evaluate a single parameter combination."""
        try:
            # Create strategy instance with these parameters
            strategy = self.strategy_class(**params)

            # Run backtest
            backtester = Backtester(
                strategy=strategy,
                data=data,
                transaction_costs=self.transaction_cost,
                benchmark=benchmark_data,
            )
            results = backtester.run()

            # Get metric value
            metrics = results.summary()
            score = metrics.get(self.metric, np.nan)

            # Handle max_drawdown (we want to minimize it, so negate)
            if self.metric == 'max_drawdown':
                score = -score  # Convert to positive for maximization

            return params, score

        except Exception as e:
            if self.verbose:
                warnings.warn(f'Error evaluating params {params}: {str(e)}', stacklevel=2)
            return params, np.nan

    def fit(
        self, data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None
    ) -> GridSearchResult:
        """
        Run grid search optimization.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with DatetimeIndex
        benchmark_data : pd.DataFrame, optional
            Benchmark data for calculating alpha and information ratio

        Returns
        -------
        GridSearchResult
            Object containing best parameters, best score, and all results
        """
        results = []

        if self.n_jobs == 1:
            # Sequential execution
            for i, params in enumerate(self.param_combinations):
                if self.verbose and (i + 1) % 10 == 0:
                    logger.debug(
                        f'Evaluating combination {i + 1}/{len(self.param_combinations)}...'
                    )

                _, score = self._evaluate_params(params, data, benchmark_data)
                results.append({'params': params, 'score': score})
        else:
            # Parallel execution
            if self.verbose:
                logger.info(f'Running grid search with {self.n_jobs} parallel jobs...')

            with ProcessPoolExecutor(
                max_workers=self.n_jobs if self.n_jobs > 0 else None
            ) as executor:
                futures = {
                    executor.submit(self._evaluate_params, params, data, benchmark_data): params
                    for params in self.param_combinations
                }

                for i, future in enumerate(as_completed(futures)):
                    if self.verbose and (i + 1) % 10 == 0:
                        logger.debug(
                            f'Completed {i + 1}/{len(self.param_combinations)} combinations...'
                        )

                    params, score = future.result()
                    results.append({'params': params, 'score': score})

        # Create results DataFrame
        results_df = pd.DataFrame([{**r['params'], self.metric: r['score']} for r in results])

        # Remove NaN results
        results_df = results_df.dropna(subset=[self.metric])

        if len(results_df) == 0:
            raise RuntimeError(
                'All parameter combinations failed. Please check your strategy and data.'
            )

        # Find best parameters
        best_idx = results_df[self.metric].idxmax()
        best_score = results_df.loc[best_idx, self.metric]

        param_cols = list(self.param_grid.keys())
        best_params = results_df.loc[best_idx, param_cols].to_dict()

        # Sort results by score
        results_df = results_df.sort_values(self.metric, ascending=False).reset_index(drop=True)

        if self.verbose:
            logger.info('Grid search complete!')
            logger.info(f'Best {self.metric}: {best_score:.4f}')
            logger.info(f'Best parameters: {best_params}')

        return GridSearchResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results_df,
            metric_name=self.metric,
        )

    def fit_with_validation(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        benchmark_validation: Optional[pd.DataFrame] = None,
    ) -> Tuple[GridSearchResult, pd.DataFrame]:
        """
        Run grid search with separate training and validation sets.

        This helps detect overfitting by showing performance on unseen data.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training OHLCV data
        validation_data : pd.DataFrame
            Validation OHLCV data
        benchmark_data : pd.DataFrame, optional
            Benchmark data for training period
        benchmark_validation : pd.DataFrame, optional
            Benchmark data for validation period

        Returns
        -------
        GridSearchResult
            Results from training data optimization
        pd.DataFrame
            Validation scores for all parameter combinations
        """
        # Run grid search on training data
        train_results = self.fit(train_data, benchmark_data)

        if self.verbose:
            logger.info('Evaluating on validation set...')

        # Evaluate all combinations on validation data
        validation_scores = []
        for params in self.param_combinations:
            _, score = self._evaluate_params(params, validation_data, benchmark_validation)
            validation_scores.append({'params': params, 'validation_score': score})

        # Create validation DataFrame
        validation_df = pd.DataFrame(
            [
                {**v['params'], f'validation_{self.metric}': v['validation_score']}
                for v in validation_scores
            ]
        )

        # Merge with training results
        param_cols = list(self.param_grid.keys())
        merged = train_results.all_results.merge(validation_df, on=param_cols, how='left')

        # Calculate overfitting metric (train - validation)
        merged['overfit_gap'] = merged[self.metric] - merged[f'validation_{self.metric}']

        if self.verbose:
            best_params = train_results.best_params
            # Find validation score for best params
            mask = True
            for k, v in best_params.items():
                mask &= merged[k] == v
            val_score = merged.loc[mask, f'validation_{self.metric}'].values[0]

            logger.info(f'Best params validation score: {val_score:.4f}')
            logger.info(f'Overfitting gap: {train_results.best_score - val_score:.4f}')

        return train_results, merged


def quick_optimize(
    strategy_class: type,
    data: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    metric: str = 'sharpe_ratio',
    top_n: int = 5,
) -> None:
    """
    Quick helper function for grid search optimization with pretty output.

    Parameters
    ----------
    strategy_class : type
        Strategy class to optimize
    data : pd.DataFrame
        OHLCV data
    param_grid : Dict[str, List[Any]]
        Parameter grid to search
    metric : str
        Metric to optimize
    top_n : int
        Number of top results to display
    """
    grid_search = GridSearch(strategy_class, param_grid, metric=metric)
    result = grid_search.fit(data)

    logger.info(result.summary())
    logger.info(f'Top {top_n} Parameter Combinations:')
    logger.info(result.get_top_n(top_n).to_string(index=False))
