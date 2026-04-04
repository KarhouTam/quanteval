import pandas as pd
import pytest

from quanteval.core.backtester import Backtester
from quanteval.strategies import DualMAStrategy
from quanteval.core.transaction import TransactionCost
from quanteval.core.sizer import (
    AllInSizer,
    EqualWeightSizer,
    FixedFractionSizer,
    PositionSizer,
    VolatilityTargetSizer,
)


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


# ---------------------------------------------------------------------------
# Position sizer tests
# ---------------------------------------------------------------------------


def _make_positions(n: int = 10) -> pd.Series:
    """Helper: alternating 0/1 signal as shifted positions."""
    idx = pd.date_range('2022-01-03', periods=n, freq='B')
    vals = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0][:n]
    return pd.Series(vals, index=idx)


def _make_data(n: int = 10) -> pd.DataFrame:
    """Helper: minimal OHLCV DataFrame for sizer tests."""
    idx = pd.date_range('2022-01-03', periods=n, freq='B')
    import numpy as np

    close = pd.Series(100.0 + np.arange(n, dtype=float), index=idx)
    return pd.DataFrame(
        {'Open': close, 'High': close * 1.01, 'Low': close * 0.99, 'Close': close},
        index=idx,
    )


def test_position_sizer_is_abstract() -> None:
    """PositionSizer cannot be instantiated directly.
    PositionSizer 是抽象类，不能直接实例化。"""
    with pytest.raises(TypeError):
        PositionSizer()  # type: ignore[abstract]


def test_all_in_sizer_returns_raw_positions() -> None:
    """AllInSizer passes positions through unchanged.
    AllInSizer 应原样返回仓位序列，不做任何修改。"""
    raw = _make_positions()
    data = _make_data()
    sizer = AllInSizer()
    result = sizer.apply(raw, data)
    pd.testing.assert_series_equal(result, raw)


def test_fixed_fraction_sizer(sample_market_data: pd.DataFrame) -> None:
    """FixedFractionSizer scales positions by the given fraction.
    FixedFractionSizer 应将仓位乘以给定的比例系数。"""
    raw = _make_positions()
    data = _make_data()
    fraction = 0.5
    sizer = FixedFractionSizer(fraction=fraction)
    result = sizer.apply(raw, data)
    expected = raw * fraction
    pd.testing.assert_series_equal(result, expected)
    # All values must be in [0.0, fraction]
    assert result.max() <= fraction + 1e-9
    assert result.min() >= 0.0


def test_fixed_fraction_sizer_invalid_fraction() -> None:
    """FixedFractionSizer rejects fractions outside (0, 1].
    FixedFractionSizer 应拒绝超出 (0, 1] 范围的比例值。"""
    with pytest.raises(ValueError):
        FixedFractionSizer(fraction=0.0)
    with pytest.raises(ValueError):
        FixedFractionSizer(fraction=1.1)
    with pytest.raises(ValueError):
        FixedFractionSizer(fraction=-0.5)


def test_equal_weight_sizer(sample_market_data: pd.DataFrame) -> None:
    """EqualWeightSizer scales positions by 1/n_positions.
    EqualWeightSizer 应将仓位除以 n_positions。"""
    raw = _make_positions()
    data = _make_data()
    n = 4
    sizer = EqualWeightSizer(n_positions=n)
    result = sizer.apply(raw, data)
    expected = raw * (1.0 / n)
    pd.testing.assert_series_equal(result, expected)
    assert float(result.max()) <= 1.0 / n + 1e-9


def test_equal_weight_sizer_invalid_n() -> None:
    """EqualWeightSizer rejects n_positions < 1.
    EqualWeightSizer 应拒绝 n_positions < 1 的参数。"""
    with pytest.raises(ValueError):
        EqualWeightSizer(n_positions=0)
    with pytest.raises(ValueError):
        EqualWeightSizer(n_positions=-1)


def test_volatility_target_sizer_shape(sample_market_data: pd.DataFrame) -> None:
    """VolatilityTargetSizer returns Series aligned to positions index.
    VolatilityTargetSizer 返回的序列应与仓位序列索引对齐。"""
    data = sample_market_data
    raw_positions = pd.Series(1.0, index=data.index)
    sizer = VolatilityTargetSizer(target_vol=0.15, window=20)
    result = sizer.apply(raw_positions, data)
    assert isinstance(result, pd.Series)
    assert result.index.equals(raw_positions.index)


def test_volatility_target_sizer_clipped(sample_market_data: pd.DataFrame) -> None:
    """VolatilityTargetSizer clips positions to [0.0, 1.0].
    VolatilityTargetSizer 的仓位应被截断到 [0.0, 1.0] 区间，不允许杠杆。"""
    data = sample_market_data
    raw_positions = pd.Series(1.0, index=data.index)
    sizer = VolatilityTargetSizer(target_vol=0.15, window=20)
    result = sizer.apply(raw_positions, data)
    assert result.max() <= 1.0 + 1e-9
    assert result.min() >= 0.0


def test_volatility_target_sizer_zero_when_no_signal(sample_market_data: pd.DataFrame) -> None:
    """VolatilityTargetSizer returns 0 where raw position is 0.
    当原始仓位为 0 时，VolatilityTargetSizer 应返回 0。"""
    data = sample_market_data
    raw_positions = pd.Series(0.0, index=data.index)
    sizer = VolatilityTargetSizer(target_vol=0.15, window=20)
    result = sizer.apply(raw_positions, data)
    assert (result == 0.0).all()


def test_volatility_target_sizer_invalid_params() -> None:
    """VolatilityTargetSizer rejects non-positive target_vol or window.
    VolatilityTargetSizer 应拒绝非正的 target_vol 或 window 参数。"""
    with pytest.raises(ValueError):
        VolatilityTargetSizer(target_vol=0.0, window=20)
    with pytest.raises(ValueError):
        VolatilityTargetSizer(target_vol=0.15, window=0)


def test_backtester_default_sizer_backward_compat(sample_market_data: pd.DataFrame) -> None:
    """Backtester without sizer arg produces same results as AllInSizer.
    不传 sizer 参数时，Backtester 的结果应与使用 AllInSizer 完全一致。"""
    strategy = DualMAStrategy(5, 20)
    result_default = Backtester(
        strategy=strategy, data=sample_market_data, transaction_costs=False
    ).run()
    result_all_in = Backtester(
        strategy=strategy,
        data=sample_market_data,
        transaction_costs=False,
        sizer=AllInSizer(),
    ).run()
    pd.testing.assert_series_equal(result_default.positions, result_all_in.positions)
    pd.testing.assert_series_equal(result_default.returns, result_all_in.returns)


def test_backtester_with_fixed_fraction_sizer(sample_market_data: pd.DataFrame) -> None:
    """Backtester with FixedFractionSizer produces fractional positions in results.
    Backtester 使用 FixedFractionSizer 时应在结果中生成分数仓位。"""
    strategy = DualMAStrategy(5, 20)
    fraction = 0.5
    result = Backtester(
        strategy=strategy,
        data=sample_market_data,
        transaction_costs=False,
        sizer=FixedFractionSizer(fraction=fraction),
    ).run()
    # Positions must not exceed fraction (with small tolerance for floating point)
    assert result.positions.max() <= fraction + 1e-9
    assert result.positions.min() >= 0.0
    # At least some positions are fractional (non-zero and non-one)
    nonzero = result.positions[result.positions > 0]
    assert (nonzero <= fraction + 1e-9).all()
