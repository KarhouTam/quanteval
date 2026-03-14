"""
Technical Factors - 技术指标因子库
Collection of technical indicators as factors.
"""

import pandas as pd
from quanteval.factors.base import Factor
from typing import Dict, Union


class SMA(Factor):
    """
    简单移动平均线 (Simple Moving Average)

    Args:
        window: 窗口期 (Window size)
    """

    def __init__(self, window: int = 20):
        super().__init__(name=f'SMA_{window}', window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Compute SMA over the Close column. Returns a pd.Series of rolling mean."""
        return data['Close'].rolling(window=self.params['window']).mean()


class EMA(Factor):
    """
    指数移动平均线 (Exponential Moving Average)

    Args:
        span: EMA周期 (EMA span)
    """

    def __init__(self, span: int = 20):
        super().__init__(name=f'EMA_{span}', span=span)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Compute EMA over the Close column. Returns a pd.Series of exponential mean."""
        return data['Close'].ewm(span=self.params['span'], adjust=False).mean()


class RSI(Factor):
    """
    相对强弱指标 (Relative Strength Index)

    Args:
        period: RSI周期 (RSI period, default 14)
    """

    def __init__(self, period: int = 14):
        super().__init__(name=f'RSI_{period}', period=period)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Compute RSI over the Close column. Returns a pd.Series of RSI values (0-100)."""
        period = self.params['period']

        # Calculate price changes
        delta = data['Close'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


class MACD(Factor):
    """
    指数平滑异同移动平均线 (Moving Average Convergence Divergence)

    Args:
        fast: 快线周期 (Fast period, default 12)
        slow: 慢线周期 (Slow period, default 26)
        signal: 信号线周期 (Signal period, default 9)
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(name=f'MACD_{fast}_{slow}_{signal}', fast=fast, slow=slow, signal=signal)

    def calculate(self, data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        Returns:
            Dictionary with 'macd', 'signal', 'histogram' keys
        """
        fast = self.params['fast']
        slow = self.params['slow']
        signal_period = self.params['signal']

        # Calculate EMAs
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()

        # MACD line (DIF)
        macd_line = ema_fast - ema_slow

        # Signal line (DEA)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Histogram (MACD Bar)
        histogram = (macd_line - signal_line) * 2

        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}


class BollingerBands(Factor):
    """
    布林带 (Bollinger Bands)

    Args:
        window: 窗口期 (Window size, default 20)
        num_std: 标准差倍数 (Number of standard deviations, default 2)
    """

    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(name=f'BB_{window}_{num_std}', window=window, num_std=num_std)

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Returns:
            Dictionary with 'middle', 'upper', 'lower' keys
        """
        window = self.params['window']
        num_std = self.params['num_std']

        # Middle band (SMA)
        middle = data['Close'].rolling(window=window).mean()

        # Standard deviation
        std = data['Close'].rolling(window=window).std()

        # Upper and lower bands
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        return {'middle': middle, 'upper': upper, 'lower': lower}


class ATR(Factor):
    """
    平均真实波幅 (Average True Range)

    Args:
        period: ATR周期 (ATR period, default 14)
    """

    def __init__(self, period: int = 14):
        super().__init__(name=f'ATR_{period}', period=period)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Compute ATR from High/Low/Close columns. Returns a pd.Series of average true range values."""
        period = self.params['period']

        # True Range components
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift(1))
        low_close = abs(data['Low'] - data['Close'].shift(1))

        # True Range is the maximum of the three
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR is the moving average of TR
        atr = tr.rolling(window=period).mean()

        return atr


class Momentum(Factor):
    """
    动量因子 (Momentum Factor)

    Calculates price change over a specified period.

    Args:
        window: 回溯窗口期 (Lookback window, default 20)
    """

    def __init__(self, window: int = 20):
        super().__init__(name=f'Momentum_{window}', window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Compute Momentum over the Close column. Returns a pd.Series of momentum values."""
        window = self.params['window']

        # Momentum = (Price / Price[t-window]) - 1
        momentum = data['Close'] / data['Close'].shift(window) - 1

        return momentum


class VolumeMA(Factor):
    """
    成交量移动平均 (Volume Moving Average)

    Args:
        window: 窗口期 (Window size, default 20)
    """

    def __init__(self, window: int = 20):
        super().__init__(name=f'VolumeMA_{window}', window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Compute Volume MA over the Volume column. Returns a pd.Series of volume moving average."""
        return data['Volume'].rolling(window=self.params['window']).mean()


class ROC(Factor):
    """
    变动率指标 (Rate of Change)

    Args:
        period: ROC周期 (ROC period, default 12)
    """

    def __init__(self, period: int = 12):
        super().__init__(name=f'ROC_{period}', period=period)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Compute ROC over the Close column. Returns a pd.Series of rate of change values."""
        period = self.params['period']

        # ROC = ((Close - Close[t-period]) / Close[t-period]) * 100
        roc = ((data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)) * 100

        return roc


class StochasticOscillator(Factor):
    """
    随机震荡指标 (Stochastic Oscillator)

    Args:
        k_period: K线周期 (K period, default 14)
        d_period: D线周期 (D period, default 3)
    """

    def __init__(self, k_period: int = 14, d_period: int = 3):
        super().__init__(name=f'Stoch_{k_period}_{d_period}', k_period=k_period, d_period=d_period)

    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        k_period = self.params['k_period']
        d_period = self.params['d_period']

        # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        lowest_low = data['Low'].rolling(window=k_period).min()
        highest_high = data['High'].rolling(window=k_period).max()

        k_line = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100

        # %D = SMA of %K
        d_line = k_line.rolling(window=d_period).mean()

        return {'k': k_line, 'd': d_line}
