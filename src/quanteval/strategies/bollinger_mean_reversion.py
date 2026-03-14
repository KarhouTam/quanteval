"""
Bollinger Bands Mean Reversion Strategy - 布林带均值回归策略
Mean reversion strategy using Bollinger Bands.
"""

import pandas as pd
import numpy as np
from quanteval.core.strategy import Strategy


class BollingerMeanReversionStrategy(Strategy):
    """
    布林带均值回归策略

    Bollinger Bands Mean Reversion Strategy.

    Strategy Logic:
        - Buy: When price touches lower band (oversold)
        - Sell: When price touches upper band (overbought)
        - Hold position until exit signal

    Args:
        window: 布林带窗口期 (Bollinger Bands window, default 20)
        num_std: 标准差倍数 (Number of standard deviations, default 2)

    Example:
        >>> strategy = BollingerMeanReversionStrategy(window=20, num_std=2)
        >>> bt = Backtester(strategy, data)
        >>> results = bt.run()
    """

    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(name=f'BollingerMR({window},{num_std})', window=window, num_std=num_std)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Generate trading signals based on Bollinger Bands.

        Returns:
            Series with values:
                1: Long position (between entry and exit)
                0: No position
        """
        window = self.params['window']
        num_std = self.params['num_std']

        # Calculate Bollinger Bands
        middle = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        # Initialize signal with NaN
        signal = pd.Series(np.nan, index=data.index)

        # Entry signal: Price touches lower band (buy signal)
        signal[data['Close'] < lower] = 1

        # Exit signal: Price touches upper band (sell signal)
        signal[data['Close'] > upper] = 0

        # Forward fill to maintain position between signals
        signal = signal.ffill()

        # Fill initial NaNs with 0 (no position)
        signal = signal.fillna(0)

        return signal
