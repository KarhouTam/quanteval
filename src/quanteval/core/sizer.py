"""
Position Sizer - 仓位计算器
Abstract base class and built-in implementations for position sizing.
"""

from abc import ABC, abstractmethod

import pandas as pd


class PositionSizer(ABC):
    """
    仓位计算器基类 - 所有仓位计算器必须继承此类

    Base class for all position sizers. Converts raw binary signals (0/1)
    into fractional position weights in [0.0, 1.0].

    The backtester calls ``apply(raw_positions, data)`` once per strategy run,
    after the look-ahead shift has already been applied to the signals.
    Subclasses must implement :meth:`apply`.
    """

    @abstractmethod
    def apply(self, raw_positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        将原始仓位信号转换为分数仓位 (必须实现)

        Convert raw shifted positions to fractional position weights.

        Args:
            raw_positions: Series of 0/1 values (already look-ahead shifted)
                           已经过前瞻偏移的原始仓位序列 (值为 0 或 1)
            data: Full OHLCV DataFrame used for context-dependent sizing
                  完整的 OHLCV 数据，供需要市场数据的仓位计算器使用

        Returns:
            Series of floats in [0.0, 1.0] representing target position fraction.
            返回 [0.0, 1.0] 范围内的浮点序列，表示目标仓位比例。
        """


class AllInSizer(PositionSizer):
    """
    全仓计算器 - 当前默认行为，信号为 1 时全仓入场

    All-in / all-out sizer. Passes raw positions through unchanged.
    This is the default behavior equivalent to the pre-sizing engine.
    """

    def apply(self, raw_positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        原样返回仓位序列

        Return raw positions unchanged (all-in behavior).
        """
        return raw_positions


class FixedFractionSizer(PositionSizer):
    """
    固定比例仓位计算器 - 信号触发时始终投入固定比例资金

    Fixed-fraction position sizer. When signal is 1, invests exactly
    ``fraction`` of portfolio value; when signal is 0, stays in cash.

    Args:
        fraction: Target position size as a fraction of portfolio, in (0.0, 1.0].
                  目标仓位占组合净值的比例，取值范围 (0.0, 1.0]。
    """

    def __init__(self, fraction: float = 0.95) -> None:
        if not (0.0 < fraction <= 1.0):
            raise ValueError(
                f'fraction must be in (0.0, 1.0], got {fraction}. 比例值必须在 (0.0, 1.0] 范围内。'
            )
        self.fraction = fraction

    def apply(self, raw_positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        将仓位乘以固定比例

        Scale positions by the fixed fraction.
        """
        return raw_positions * self.fraction


class EqualWeightSizer(PositionSizer):
    """
    等权仓位计算器 - 将资金平均分配给 n_positions 个同时持有的仓位

    Equal-weight position sizer. Divides capital equally across
    ``n_positions`` simultaneous positions.

    Useful when running multiple strategies or instruments in parallel and
    wanting each to receive 1/N of total capital.

    Args:
        n_positions: Number of simultaneous positions to size for.
                     同时持有的仓位数量，每个仓位分配 1/n_positions 比例的资金。
    """

    def __init__(self, n_positions: int = 5) -> None:
        if n_positions < 1:
            raise ValueError(
                f'n_positions must be >= 1, got {n_positions}. n_positions 必须大于等于 1。'
            )
        self.n_positions = n_positions

    def apply(self, raw_positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        将仓位除以 n_positions

        Scale positions by 1 / n_positions.
        """
        return raw_positions * (1.0 / self.n_positions)


class VolatilityTargetSizer(PositionSizer):
    """
    波动率目标仓位计算器 - 根据滚动波动率动态调整仓位

    Volatility-targeting position sizer. Scales position size inversely
    proportional to recent rolling volatility so that the strategy targets
    a constant annualised volatility contribution.

    Position fraction = min(target_vol / rolling_std, 1.0)

    When rolling volatility is high the position is reduced; when volatility
    is low the position is increased (up to a maximum of 1.0, no leverage).

    Args:
        target_vol: Target annualised volatility (e.g. 0.15 = 15% p.a.).
                    目标年化波动率（例如 0.15 表示 15%）。
        window: Rolling window in trading days for volatility estimation.
                计算滚动波动率所用的交易日窗口长度。
    """

    def __init__(self, target_vol: float = 0.15, window: int = 20) -> None:
        if target_vol <= 0.0:
            raise ValueError(f'target_vol must be > 0, got {target_vol}. target_vol 必须大于 0。')
        if window < 1:
            raise ValueError(f'window must be >= 1, got {window}. window 必须大于等于 1。')
        self.target_vol = target_vol
        self.window = window

    def apply(self, raw_positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        根据滚动波动率调整仓位，超过 1.0 时截断（不允许杠杆）

        Scale positions by target_vol / rolling_std, clipped to [0.0, 1.0].
        Bars with insufficient history (NaN volatility) receive a position of 0.
        """
        daily_vol: pd.Series = (
            data['Close'].pct_change().rolling(self.window).std().reindex(raw_positions.index)
        )
        size_factor: pd.Series = (self.target_vol / daily_vol).clip(lower=0.0, upper=1.0)
        return (raw_positions * size_factor).fillna(0.0)
