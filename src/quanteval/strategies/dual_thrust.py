"""
Dual Thrust Strategy -
Strategy Logic:
    - Buy: When price breaks above the upper threshold (based on previous range)
    - Sell: When price breaks below the lower threshold (based on previous range)
策略逻辑：
    - 买入：当价格突破上轨（基于前一天的区间）
    - 卖出：当价格突破下轨（基于前一天的区间）
"""

import pandas as pd
import numpy as np
from quanteval.core.strategy import Strategy


class DualThrustStrategy(Strategy):
    """
    Dual Thrust Strategy - A breakout strategy based on price range.

    双重突破策略，基于价格区间的突破策略。

    Strategy Logic:
        - Buy: When price breaks above the upper threshold (based on previous range)
        - Sell: When price breaks below the lower threshold (based on previous range)
    策略逻辑：
        - 买入：当价格突破上轨（基于前一天的区间）
        - 卖出：当价格突破下轨（基于前一天的区间）
    Args:
        k1: 上轨系数 (Upper threshold multiplier, default 0.5)
        k2: 下轨系数 (Lower threshold multiplier, default 0.5)

    Example:
        >>> strategy = DualThrustStrategy(k1=0.5, k2=0.5)
        >>> bt = Backtester(strategy, data)
        >>> results = bt.run()
    """

    def __init__(self, k1: float = 0.5, k2: float = 0.5, window: int = 5):
        super().__init__(
            name=f'DualThrust(k1={k1},k2={k2}, N={window})', k1=k1, k2=k2, window=window
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Generate trading signals based on Dual Thrust logic.

        Returns:
            Series with values:
                1: Long position (price > upper threshold)
                0: No position (price < lower threshold)
        """
        k1 = self.params['k1']
        k2 = self.params['k2']
        n = self.params['window']

        # Calculate previous day's range
        hh = data['High'].shift(1).rolling(window=n).max()
        hc = data['Close'].shift(1).rolling(window=n).max()
        ll = data['Low'].shift(1).rolling(window=n).min()
        lc = data['Close'].shift(1).rolling(window=n).min()

        diff = np.maximum(hh - lc, hc - ll)

        # Calculate thresholds
        upper = data['Open'] + k1 * diff
        lower = data['Open'] - k2 * diff

        # Generate signals
        signal = pd.Series(np.nan, index=data.index, name='Signal')
        signal[data['Close'] > upper] = 1
        signal[data['Close'] < lower] = 0

        return signal.ffill().fillna(0)
