import pandas as pd

from quanteval.strategies import BollingerMeanReversionStrategy, DualMAStrategy
from quanteval.utils.comparison import StrategyComparator


def test_strategy_comparison_builds_metrics_table(sample_market_data: pd.DataFrame) -> None:
    comparator = StrategyComparator(
        [DualMAStrategy(5, 20), BollingerMeanReversionStrategy(20, 2.0)],
        verbose=False,
    )
    result = comparator.compare(sample_market_data)

    assert result.metrics_df.shape[0] == 2
    assert 'sharpe_ratio' in result.metrics_df.columns


def test_equal_weight_portfolio_returns_backtest_result(sample_market_data: pd.DataFrame) -> None:
    comparator = StrategyComparator(
        [DualMAStrategy(5, 20), BollingerMeanReversionStrategy(20, 2.0)],
        verbose=False,
    )
    portfolio = comparator.create_equal_weight_portfolio(sample_market_data)

    assert isinstance(portfolio.summary(), pd.Series)
    assert portfolio.strategy_name == 'Portfolio'
