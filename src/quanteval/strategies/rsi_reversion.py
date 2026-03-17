"""
RSI Reversion Strategy - RSI 反转策略
Classic mean-reversion strategy using RSI overbought/oversold signals.
"""

import pandas as pd
import numpy as np
from quanteval.core.strategy import Strategy
from quanteval.factors import RSI


class RSIStrategy(Strategy):
    """
    RSI 反转策略 - 经典均值回归策略

    RSI Mean Reversion Strategy.

    Strategy Logic:
        - Buy: When RSI < oversold (market oversold)
        - Sell: When RSI > overbought (market overbought)

    Args:
        window: RSI 计算窗口 (RSI window, default 14)
        oversold: 超卖阈值 (Oversold level, default 30)
        overbought: 超买阈值 (Overbought level, default 70)

    Example:
        >>> strategy = RSIStrategy(window=14, oversold=30, overbought=70)
        >>> bt = Backtester(strategy, data)
        >>> results = bt.run()
    """

    def __init__(self, window: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(
            name=f'RSI({window})',
            window=window,
            oversold=oversold,
            overbought=overbought,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Generate trading signals based on RSI levels.

        Returns:
            Series with values:
                1: Long position (RSI < oversold)
                0: No position (RSI >= oversold)
        """

        window = self.params['window']
        oversold = self.params['oversold']
        overbought = self.params['overbought']

        rsi = RSI(period=window).calculate(data)

        signal = pd.Series(np.nan, index=data.index, name='Signal')
        signal[rsi < oversold] = 1
        signal[rsi > overbought] = 0

        return signal.ffill().fillna(0)
