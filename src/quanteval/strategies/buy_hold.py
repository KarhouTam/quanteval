"""
Buy and Hold Strategy - 买入持有策略
Simple baseline strategy that buys and holds.
"""

import pandas as pd
from quanteval.core.strategy import Strategy


class BuyAndHoldStrategy(Strategy):
    """
    买入持有策略 - 基准策略

    Buy and Hold Strategy - baseline for comparison.

    Simply buys at the start and holds until the end.

    Example:
        >>> strategy = BuyAndHoldStrategy()
        >>> bt = Backtester(strategy, data)
        >>> results = bt.run()
    """

    def __init__(self):
        super().__init__(name='BuyAndHold')

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号 - 始终持有

        Generate signals - always hold.

        Returns:
            Series of 1s (always long)
        """
        # Always hold position
        signal = pd.Series(1, index=data.index, name='Signal')

        return signal
