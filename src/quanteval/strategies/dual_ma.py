"""
Dual Moving Average Strategy - 双均线策略
Classic trend-following strategy using moving average crossovers.
"""

import pandas as pd
import numpy as np
from quanteval.core.strategy import Strategy


class DualMAStrategy(Strategy):
    """
    双均线策略 - 经典趋势跟踪策略

    Dual Moving Average Crossover Strategy.

    Strategy Logic:
        - Buy (Golden Cross): When fast MA crosses above slow MA
        - Sell (Death Cross): When fast MA crosses below slow MA

    Args:
        fast_window: 快线窗口期 (Fast MA window, default 10)
        slow_window: 慢线窗口期 (Slow MA window, default 60)
        ma_type: 均线类型 'sma' or 'ema' (MA type, default 'sma')

    Example:
        >>> strategy = DualMAStrategy(fast_window=10, slow_window=60)
        >>> bt = Backtester(strategy, data)
        >>> results = bt.run()
    """

    def __init__(self, fast_window: int = 10, slow_window: int = 60, ma_type: str = 'sma'):
        super().__init__(
            name=f'DualMA({fast_window},{slow_window})',
            fast_window=fast_window,
            slow_window=slow_window,
            ma_type=ma_type,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Generate trading signals based on MA crossover.

        Returns:
            Series with values:
                1: Long position (fast MA > slow MA)
                0: No position (fast MA <= slow MA)
        """
        fast_window = self.params['fast_window']
        slow_window = self.params['slow_window']
        ma_type = self.params['ma_type']

        # Calculate moving averages
        if ma_type == 'ema':
            fast_ma = data['Close'].ewm(span=fast_window, adjust=False).mean()
            slow_ma = data['Close'].ewm(span=slow_window, adjust=False).mean()
        else:  # sma
            fast_ma = data['Close'].rolling(window=fast_window).mean()
            slow_ma = data['Close'].rolling(window=slow_window).mean()

        # Generate signals
        signal = pd.Series(np.nan, index=data.index, name='Signal')
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma <= slow_ma] = 0

        return signal.ffill().fillna(0)
