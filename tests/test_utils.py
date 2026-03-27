import pandas as pd

import pytest

from quanteval.utils import align_series, calculate_cumulative_returns, calculate_returns


def test_calculate_returns_and_cumulative_returns() -> None:
    prices = pd.Series([100.0, 110.0, 121.0])
    returns = calculate_returns(prices)
    cumulative = calculate_cumulative_returns(returns.fillna(0.0))

    assert returns.iloc[1] == pytest.approx(0.1)
    assert cumulative.iloc[-1] == pytest.approx(1.21)


def test_align_series_keeps_common_index() -> None:
    left = pd.Series([1, 2, 3], index=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']))
    right = pd.Series([10, 20], index=pd.to_datetime(['2020-01-02', '2020-01-03']))

    aligned_left, aligned_right = align_series(left, right)

    assert len(aligned_left) == 2
    assert aligned_left.index.equals(aligned_right.index)
