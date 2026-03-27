import numpy as np
import pandas as pd
import pytest

from quanteval.strategies.multi_factor_model import MultiFactorModel
from quanteval.aggregators.weighted import (
	EqualWeightAggregator,
	ICWeightedAggregator,
)
from quanteval.factors.base import Factor


def test_equal_weight_aggregator_without_normalize(sample_market_data: pd.DataFrame) -> None:
	data = sample_market_data
	idx = data.index
	s1 = data['Close'].pct_change().fillna(0)
	s2 = -s1

	agg = EqualWeightAggregator(normalize=False)
	composite = agg.aggregate({'s1': s1, 's2': s2}, data)

	expected = (s1.reindex(idx) + s2.reindex(idx)) / 2
	pd.testing.assert_series_equal(composite, expected, check_names=False)


def test_icweighted_fallback_to_equal_weights(sample_market_data: pd.DataFrame) -> None:
	# Force forward returns to be NaN so IC is unavailable and aggregator falls
	# back to equal weights.
	data = sample_market_data.copy()
	data['Close'] = np.nan
	idx = data.index

	s1 = pd.Series(np.arange(len(idx)), index=idx, name='s1')
	s2 = pd.Series(np.arange(len(idx)) * 2.0, index=idx, name='s2')

	agg = ICWeightedAggregator(normalize=False, ic_window=3, min_periods=1)
	composite = agg.aggregate({'s1': s1, 's2': s2}, data)

	expected = (s1 + s2) / 2
	pd.testing.assert_series_equal(composite, expected, check_names=False)


class ConstantFactor(Factor):
	def __init__(self, name: str, series: pd.Series):
		super().__init__(name=name)
		self._series = series

	def calculate(self, data: pd.DataFrame) -> pd.Series:
		return self._series.reindex(data.index)


def test_multi_factor_model_signals_and_type_error(sample_market_data: pd.DataFrame) -> None:
	data = sample_market_data.iloc[:4]
	idx = data.index

	s = pd.Series([0.0, 1.0, 0.0, 1.0], index=idx)
	f1 = ConstantFactor('f1', s)
	f2 = ConstantFactor('f2', s)

	agg = EqualWeightAggregator(normalize=False)
	model = MultiFactorModel([f1, f2], aggregator=agg, threshold=0.5)

	signals = model.generate_signals(data)
	expected = (((s + s) / 2) > 0.5).astype(float).ffill().fillna(0.0)
	expected.name = signals.name
	pd.testing.assert_series_equal(signals, expected)

	# Factor returning a dict should raise TypeError
	class DictFactor(Factor):
		def __init__(self):
			super().__init__(name='dict')

		def calculate(self, data: pd.DataFrame):
			return {'a': pd.Series([1] * len(data), index=data.index)}

	bad = DictFactor()
	model2 = MultiFactorModel([bad], aggregator=agg)
	with pytest.raises(TypeError):
		model2.generate_signals(data)

