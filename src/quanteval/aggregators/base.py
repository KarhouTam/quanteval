from abc import ABC, abstractmethod

import pandas as pd


class FactorAggregator(ABC):
    """
    因子聚合器基类

    Abstract base class for factor aggregation. All custom aggregators must
    inherit from this class.

    Args:
        name: 聚合器名称 (Aggregator name)
    """

    def __init__(self, name: str = 'FactorAggregator'):
        self.name = name

    @abstractmethod
    def aggregate(self, factors: dict[str, pd.Series], data: pd.DataFrame) -> pd.Series:
        """
        聚合多个因子为复合信号 (必须实现)

        Aggregate multiple factors into a composite score. Must be implemented
        by subclasses.

        Args:
            factors: Dict mapping factor names to their Series values
            data: Original OHLCV DataFrame (used for IC calculation in subclasses)

        Returns:
            Composite score Series with same index as data
        """
        pass

    def __repr__(self) -> str:
        return self.name
