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


class HKTransactionCost(TransactionCost):
    """
    Transaction cost model for Hong Kong (HKEX) market.

    Key differences from A-share (TransactionCost):
    - Stamp duty is BILATERAL: applies on both buy and sell (0.10% each side, since Aug 2023).
      A-share stamp duty is sell-side only.
    - SFC transaction levy (0.0027% bilateral) and HKEX trading fee (0.00565% bilateral)
      are additional regulatory charges not present in the A-share model.
    - Commission and slippage remain bilateral, same as A-share.

    Default rates reflect Hong Kong market rules as of 2026:
        - Commission: 0.05% (both sides, typical HK retail)
        - Stamp duty: 0.10% (both sides, reduced from 0.26% on 16 Aug 2023)
        - Slippage: 0.05% (both sides)
        - SFC levy: 0.0027% (both sides)
        - HKEX trading fee: 0.00565% (both sides)
        - Min commission: 0 (no minimum enforced by default)
    """

    def __init__(
        self,
        commission_rate: float = 0.0005,
        stamp_duty_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        sfc_levy_rate: float = 0.000027,
        hkex_trading_fee_rate: float = 0.0000565,
        min_commission: float = 0.0,
    ):
        super().__init__(
            commission_rate=commission_rate,
            stamp_duty_rate=stamp_duty_rate,
            slippage_rate=slippage_rate,
            min_commission=min_commission,
        )
        self.sfc_levy_rate = sfc_levy_rate
        self.hkex_trading_fee_rate = hkex_trading_fee_rate

    def calculate(
        self, positions: pd.Series, prices: pd.Series, initial_capital: float = 100000.0
    ) -> pd.Series:
        """
        Calculate transaction costs for HK market (bilateral stamp duty).

        Args:
            positions: Series of positions (0 or 1)
            prices: Series of prices (Close price)
            initial_capital: Initial capital for minimum commission calculation

        Returns:
            Series of daily transaction costs (as proportion of capital)
        """
        if initial_capital <= 0:
            raise ValueError('initial_capital must be positive')

        position_diff = positions.diff().fillna(0.0)

        # trade_mask fires on ANY trade (buy or sell) — bilateral for HK
        trade_mask = (position_diff.abs() > 0).astype(float)
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

        # Stamp duty is BILATERAL in HK (both buy and sell)
        stamp_duty_cost = trade_mask * self.stamp_duty_rate

        # Regulatory charges: SFC levy + HKEX trading fee (both bilateral)
        levy_cost = trade_mask * (self.sfc_levy_rate + self.hkex_trading_fee_rate)

        return commission_cost + slippage_cost + stamp_duty_cost + levy_cost

    def total_cost_rate(self, trade_type: str = 'round_trip') -> float:
        """
        Calculate total cost rate for HK market.

        HK costs are symmetric (buy == sell) due to bilateral stamp duty.

        Args:
            trade_type: 'buy', 'sell', or 'round_trip'

        Returns:
            Total cost rate
        """
        if trade_type == 'buy':
            return (
                self.commission_rate
                + self.stamp_duty_rate
                + self.slippage_rate
                + self.sfc_levy_rate
                + self.hkex_trading_fee_rate
            )
        elif trade_type == 'sell':
            # Identical to buy — HK stamp duty is bilateral
            return (
                self.commission_rate
                + self.stamp_duty_rate
                + self.slippage_rate
                + self.sfc_levy_rate
                + self.hkex_trading_fee_rate
            )
        elif trade_type == 'round_trip':
            return 2 * self.total_cost_rate('buy')
        else:
            raise ValueError("trade_type must be 'buy', 'sell', or 'round_trip'")

    def __repr__(self) -> str:
        return (
            f'HKTransactionCost(commission={self.commission_rate:.4%}, '
            f'stamp_duty={self.stamp_duty_rate:.4%}, '
            f'slippage={self.slippage_rate:.4%}, '
            f'round_trip={self.total_cost_rate("round_trip"):.4%})'
        )
