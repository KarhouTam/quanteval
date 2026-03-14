"""
Transaction Cost Model - 交易成本模型
Models realistic transaction costs for A-share market.
"""

import pandas as pd
import numpy as np


class TransactionCost:
    """
    交易成本模型 - A股市场交易成本

    Transaction cost model for Chinese A-share market.

    Args:
        commission_rate: 佣金费率 (Commission rate, both buy and sell)
        stamp_duty_rate: 印花税率 (Stamp duty rate, sell-side only)
        slippage_rate: 滑点费率 (Slippage rate, both sides)
        min_commission: 最小佣金 (Minimum commission per trade)

    Default A-share costs:
        - Commission: 0.03% (both sides)
        - Stamp duty: 0.1% (sell-side only)
        - Slippage: 0.1% (both sides)
        - Min commission: 5 CNY (usually negligible for large trades)
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,
        stamp_duty_rate: float = 0.001,
        slippage_rate: float = 0.001,
        min_commission: float = 0.0,
    ):
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage_rate = slippage_rate
        self.min_commission = min_commission

    def calculate(
        self, positions: pd.Series, prices: pd.Series, initial_capital: float = 100000.0
    ) -> pd.Series:
        """
        计算交易成本

        Calculate transaction costs based on position changes.

        Args:
            positions: Series of positions (0 or 1)
            prices: Series of prices (Close price)
            initial_capital: Initial capital for calculating shares

        Returns:
            Series of daily transaction costs (as proportion of capital)
        """
        if initial_capital <= 0:
            raise ValueError('initial_capital must be positive')

        # Detect trades (position changes)
        position_diff = positions.diff().fillna(0.0)

        # Buy signal: 0 -> 1 (position_diff = 1)
        # Sell signal: 1 -> 0 (position_diff = -1)

        sell_mask = position_diff == -1

        trade_scale = position_diff.abs().clip(lower=0.0)

        commission_cost = trade_scale * self.commission_rate
        if self.min_commission > 0:
            min_commission_rate = self.min_commission / initial_capital
            commission_cost = np.where(
                trade_scale > 0,
                np.maximum(commission_cost, min_commission_rate),
                0.0,
            )
            commission_cost = pd.Series(commission_cost, index=positions.index)
        else:
            commission_cost = pd.Series(commission_cost, index=positions.index)

        slippage_cost = trade_scale * self.slippage_rate
        stamp_duty_cost = sell_mask.astype(float) * self.stamp_duty_rate

        costs = commission_cost + slippage_cost + stamp_duty_cost

        return costs

    def total_cost_rate(self, trade_type: str = 'round_trip') -> float:
        """
        计算总成本率

        Calculate total cost rate.

        Args:
            trade_type: 'buy', 'sell', or 'round_trip'

        Returns:
            Total cost rate
        """
        if trade_type == 'buy':
            return self.commission_rate + self.slippage_rate
        elif trade_type == 'sell':
            return self.commission_rate + self.stamp_duty_rate + self.slippage_rate
        elif trade_type == 'round_trip':
            return self.commission_rate * 2 + self.stamp_duty_rate + self.slippage_rate * 2
        else:
            raise ValueError("trade_type must be 'buy', 'sell', or 'round_trip'")

    def __repr__(self) -> str:
        return (
            f'TransactionCost(commission={self.commission_rate:.4%}, '
            f'stamp_duty={self.stamp_duty_rate:.4%}, '
            f'slippage={self.slippage_rate:.4%}, '
            f'round_trip={self.total_cost_rate("round_trip"):.4%})'
        )
