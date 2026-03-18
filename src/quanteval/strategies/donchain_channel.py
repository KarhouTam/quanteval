"""
Donchian Channel Strategy - 唐奇安通道策略

Donchian Channel strategy based on breakout and breakdown signals from the channel.
"""

import pandas as pd
import numpy as np
from quanteval.core.strategy import Strategy


class DonchianChannel(Strategy):
    """
    唐奇安通道策略 - Donchian Channel Strategy

    Strategy Logic:
        - Buy: Close price breaks above the upper channel
        - Exit: Close price falls below the lower channel

    Args:
        period: 通道计算周期 (Channel period, default 20)
        multiplier: 通道宽度倍数 (Channel width multiplier, default 2)

    Example:
        >>> strategy = DonchainChannel(period=20, multiplier=2)
        >>> bt = Backtester(strategy, data)
        >>> results = bt.run()
    """

    def __init__(self, period: int = 20, multiplier: float = 2.0):
        super().__init__(
            name=f'DonchainChannel({period},{multiplier})',
            period=period,
            multiplier=multiplier,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Generate trading signals based on Donchian Channel breakouts.

        Returns:
            Series with values:
                1: Long position (close > upper channel)
                0: No position (close < lower channel)
        """
        period = self.params['period']
        multiplier = self.params['multiplier']

        # Channel boundaries
        highest_high = data['High'].rolling(window=period).max()
        lowest_low = data['Low'].rolling(window=period).min()
        middle_line = (highest_high + lowest_low) / 2
        upper_channel = middle_line + multiplier * (highest_high - lowest_low)
        lower_channel = middle_line - multiplier * (highest_high - lowest_low)

        signal = pd.Series(np.nan, index=data.index, name='Signal')
        signal[data['Close'] > upper_channel] = 1
        signal[data['Close'] < lower_channel] = 0

        return signal.ffill().fillna(0)
