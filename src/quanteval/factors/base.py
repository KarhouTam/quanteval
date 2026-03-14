"""
Factor Base Class - 因子基类
Abstract base class for all factors.
"""

from abc import ABC, abstractmethod
import pandas as pd


class Factor(ABC):
    """
    因子基类 - 所有因子必须继承此类

    Base class for all factors. All custom factors must inherit from this.

    Args:
        name: 因子名称 (Factor name)
        params: 因子参数 (Factor parameters)
    """

    def __init__(self, name: str = 'Factor', **params):
        self.name = name
        self.params = params

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值 (必须实现)

        Calculate factor values. Must be implemented by subclasses.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Series with factor values, same index as data
        """
        pass

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """Allow factor to be called as a function"""
        return self.calculate(data)

    def __repr__(self) -> str:
        params_str = ', '.join([f'{k}={v}' for k, v in self.params.items()])
        return f'{self.name}({params_str})'
