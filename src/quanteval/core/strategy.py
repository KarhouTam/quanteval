"""
Base Strategy Class - 策略基类
Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class Strategy(ABC):
    """
    策略基类 - 所有策略必须继承此类

    Base class for all trading strategies. All custom strategies must inherit from this.

    Args:
        name: 策略名称 (Strategy name)
        params: 策略参数字典 (Strategy parameters dictionary)
    """

    def __init__(self, name: str = 'Strategy', **params):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号 (必须实现)

        Generate trading signals. Must be implemented by subclasses.

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex

        Returns:
            Series with same index as data, values are:
                - 0: Cash / No position
                - 1: Long position

        Note:
            - Do NOT shift signals here - the backtester handles this automatically
            - Signals should be based on data available at time t
            - The backtester will apply .shift(1) to avoid look-ahead bias
            - The backtester's PositionSizer converts these binary signals into
              fractional position weights [0.0, 1.0] after the look-ahead shift.
              Backtester 中的 PositionSizer 在应用前瞻偏移后将信号转换为
              分数仓位权重 [0.0, 1.0]，策略本身仍应返回二值信号 {0, 1}。
        """
        pass

    def __repr__(self) -> str:
        params_str = ', '.join([f'{k}={v}' for k, v in self.params.items()])
        return f'{self.name}({params_str})'

    def get_params(self) -> Dict[str, Any]:
        """获取策略参数 (Get strategy parameters)"""
        return self.params.copy()

    def set_params(self, **params):
        """设置策略参数 (Set strategy parameters)"""
        self.params.update(params)
