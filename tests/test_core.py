import pandas as pd
import pytest

from quanteval.core.backtester import Backtester
from quanteval.strategies import DualMAStrategy
from quanteval.core.transaction import TransactionCost


def test_backtester_adds_returns_when_missing(sample_market_data: pd.DataFrame) -> None:
    data = sample_market_data.drop(columns=['Ret'])
    result = Backtester(strategy=DualMAStrategy(5, 20), data=data, transaction_costs=False).run()

    assert 'Ret' in result.data.columns
    assert len(result.returns) == len(data)


def test_backtester_accepts_benchmark_alias(
    sample_market_data: pd.DataFrame, sample_benchmark_data: pd.DataFrame
) -> None:
    result = Backtester(
        strategy=DualMAStrategy(5, 20),
        data=sample_market_data,
        benchmark=sample_benchmark_data,
        transaction_costs=False,
    ).run()

    summary = result.summary()
    assert 'benchmark_return' in summary
    assert result.benchmark_returns is not None


def test_transaction_cost_applies_minimum_commission() -> None:
    positions = pd.Series([0.0, 1.0, 1.0, 0.0])
    prices = pd.Series([10.0, 10.5, 10.7, 10.2])
    model = TransactionCost(
        commission_rate=0.0001, slippage_rate=0.0, stamp_duty_rate=0.0, min_commission=5.0
    )

    costs = model.calculate(positions=positions, prices=prices, initial_capital=1000.0)

    assert costs.iloc[1] == pytest.approx(0.005)
    assert costs.iloc[3] == pytest.approx(0.005)


def test_transaction_cost_rejects_non_positive_capital() -> None:
    model = TransactionCost()
    with pytest.raises(ValueError):
        model.calculate(pd.Series([0.0, 1.0]), pd.Series([10.0, 10.5]), initial_capital=0.0)


def test_hk_transaction_cost_bilateral_stamp_duty() -> None:
    from quanteval import HKTransactionCost

    positions = pd.Series([0.0, 1.0, 1.0, 0.0], index=pd.date_range('2023-01-01', periods=4))
    prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=positions.index)

    hk_tc = HKTransactionCost()
    hk_costs = hk_tc.calculate(positions, prices)

    a_tc = TransactionCost()
    a_costs = a_tc.calculate(positions, prices)

    # HK applies stamp duty on BUY (index 1, position_diff = +1)
    assert hk_costs.iloc[1] > 0
    assert a_costs.iloc[1] > 0
    assert hk_costs.iloc[1] > a_costs.iloc[1]

    # HK applies stamp duty on SELL too (index 3, position_diff = -1)
    assert hk_costs.iloc[3] > 0
    assert a_costs.iloc[3] > 0

    # HK buy cost == HK sell cost (bilateral symmetry)
    assert abs(hk_costs.iloc[1] - hk_costs.iloc[3]) < 1e-10

    # API symmetry
    assert hk_tc.total_cost_rate('buy') == hk_tc.total_cost_rate('sell')
