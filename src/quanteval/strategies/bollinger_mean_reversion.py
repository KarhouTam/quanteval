"""
Bollinger Bands Mean Reversion Strategy - 布林带均值回归策略
Mean reversion strategy using Bollinger Bands.
"""

import pandas as pd
import numpy as np
from quanteval.core.strategy import Strategy
from quanteval.factors import BollingerBands


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

        bands = BollingerBands(window=window, num_std=num_std).calculate(data)

        # Generate signals
        signal = pd.Series(np.nan, index=data.index, name='Signal')
        signal[data['Close'] < bands['lower']] = 1
        signal[data['Close'] > bands['upper']] = 0

        return signal.ffill().fillna(0)
