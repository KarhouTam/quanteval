"""
Factor Model - 因子模型
Multi-factor models combining multiple technical factors into composite trading signals.
"""

import pandas as pd

from quanteval.aggregators import EqualWeightAggregator, FactorAggregator
from quanteval.core.strategy import Strategy
from quanteval.factors.base import Factor


class MultiFactorModel(Strategy):
    """
    多因子模型基类

    Base class for multi-factor models. All custom multi-factor models must
    inherit from this class.

    Args:
        factors: 因子列表 (List of Factor instances)
        aggregator: 因子聚合器 (FactorAggregator instance, default: EqualWeightAggregator())
        threshold: 信号阈值 (Signal threshold, default: 0.0)
        name: 模型名称 (Model name)
    """

    def __init__(
        self,
        factors: list[Factor],
        aggregator: FactorAggregator | None = None,
        threshold: float = 0.0,
        name: str = 'MultiFactorModel',
    ):
        self.factors = factors
        self.aggregator = aggregator if aggregator is not None else EqualWeightAggregator()
        self.threshold = threshold
        super().__init__(name=name, threshold=threshold)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Generate trading signals based on multi-factor aggregation.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Series with values:
                1.0: Long position (composite > threshold)
                0.0: No position (composite <= threshold)

        Raises:
            TypeError: If any factor returns a dict instead of Series
        """
        factors_dict = {}

        for factor in self.factors:
            result = factor.calculate(data)

            if isinstance(result, dict):
                raise TypeError(
                    f"Factor '{factor.name}' returned a dict — extract the desired "
                    f"Series first (e.g., pass BollingerBands().calculate(data)['middle'])"
                )

            factors_dict[factor.name] = result

        composite = self.aggregator.aggregate(factors_dict, data)
        signals = (composite > self.threshold).astype(float)
        return signals.ffill().fillna(0.0)
