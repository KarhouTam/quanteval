import pandas as pd

from quanteval.strategies import DualMAStrategy
from quanteval.optimization import GridSearch, WalkForwardAnalysis


def test_grid_search_finds_best_params(sample_market_data: pd.DataFrame) -> None:
    search = GridSearch(
        DualMAStrategy,
        {'fast_window': [5, 10], 'slow_window': [20, 30], 'ma_type': ['sma']},
        verbose=False,
    )
    result = search.fit(sample_market_data)

    assert set(result.best_params) == {'fast_window', 'slow_window', 'ma_type'}
    assert not result.all_results.empty


def test_walk_forward_analysis_returns_combined_result(sample_market_data: pd.DataFrame) -> None:
    analysis = WalkForwardAnalysis(
        DualMAStrategy,
        {'fast_window': [5, 10], 'slow_window': [20, 30], 'ma_type': ['sma']},
        train_period=120,
        test_period=40,
        verbose=False,
    )
    result = analysis.run(sample_market_data)

    assert len(result.windows) > 0
    assert isinstance(result.combined_results.summary(), pd.Series)
    pd.testing.assert_series_equal(
        result.combined_results.equity_curve,
        (1.0 + result.combined_results.returns).cumprod(),
    )
