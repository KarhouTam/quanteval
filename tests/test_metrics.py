import numpy as np
import pandas as pd

from quanteval.metrics.performance import PerformanceMetrics


def test_sharpe_ratio_returns_float(sample_market_data: pd.DataFrame) -> None:
    perf = PerformanceMetrics(sample_market_data['Ret'])
    assert isinstance(perf.sharpe_ratio(), float)


def test_max_drawdown_is_non_positive(sample_market_data: pd.DataFrame) -> None:
    perf = PerformanceMetrics(sample_market_data['Ret'])
    assert perf.max_drawdown() <= 0


def test_trade_statistics_require_positions(sample_market_data: pd.DataFrame) -> None:
    perf = PerformanceMetrics(sample_market_data['Ret'])
    assert np.isnan(perf.win_rate())
    assert perf.num_trades() == 0
