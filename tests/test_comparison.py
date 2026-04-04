import pandas as pd

from quanteval.core.backtester import Backtester, BacktestResults, MultiBacktestResults
from quanteval.strategies import BollingerMeanReversionStrategy, DualMAStrategy


def test_backtester_multi_strategy_builds_comparison_tables(
    sample_market_data: pd.DataFrame,
) -> None:
    result = Backtester(
        strategy=[DualMAStrategy(5, 20), BollingerMeanReversionStrategy(20, 2.0)],
        data=sample_market_data,
        transaction_costs=False,
    ).run()

    assert isinstance(result, MultiBacktestResults)
    assert result.strategy_names == ['DualMA(5,20)', 'BollingerMR(20,2.0)']
    assert result.metrics_df.index.tolist() == result.strategy_names
    assert 'sharpe_ratio' in result.metrics_df.columns
    assert result.returns_df.columns.tolist() == result.strategy_names
    assert result.equity_df.columns.tolist() == result.strategy_names
    assert result.correlation_matrix.index.tolist() == result.strategy_names
    assert result.correlation_matrix.columns.tolist() == result.strategy_names


def test_backtester_multi_strategy_preserves_duplicate_names(
    sample_market_data: pd.DataFrame,
) -> None:
    result = Backtester(
        strategy=[DualMAStrategy(5, 20), DualMAStrategy(5, 20)],
        data=sample_market_data,
        transaction_costs=False,
    ).run()

    assert isinstance(result, MultiBacktestResults)
    assert list(result.results) == ['DualMA(5,20)', 'DualMA(5,20)_2']
    assert result.strategy_names == ['DualMA(5,20)', 'DualMA(5,20)_2']


def test_multi_backtest_results_equal_weight_portfolio(
    sample_market_data: pd.DataFrame,
) -> None:
    result = Backtester(
        strategy={
            'DualMA(5,20)': DualMAStrategy(5, 20),
            'BollingerMR': BollingerMeanReversionStrategy(20, 2.0),
        },
        data=sample_market_data,
        transaction_costs=False,
    ).run()

    assert isinstance(result, MultiBacktestResults)

    portfolio = result.create_equal_weight_portfolio()

    assert isinstance(portfolio, BacktestResults)
    assert portfolio.strategy_name == 'Portfolio'

    expected_returns = result.returns_df.mean(axis=1)
    expected_positions = pd.DataFrame(
        {name: backtest.positions for name, backtest in result.results.items()}
    ).mean(axis=1)
    pd.testing.assert_series_equal(portfolio.returns, expected_returns, check_names=False)
    pd.testing.assert_series_equal(portfolio.positions, expected_positions, check_names=False)

    summary = portfolio.summary()
    assert pd.isna(summary['num_trades'])
    assert pd.isna(summary['win_rate'])
    assert pd.isna(summary['profit_loss_ratio'])
    assert pd.isna(summary['avg_trade_duration'])
