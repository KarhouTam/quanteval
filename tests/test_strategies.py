import importlib

import pandas as pd


def test_all_strategies_return_binary_signals(sample_market_data: pd.DataFrame) -> None:
    try:
        strategies = importlib.import_module('quanteval.strategies').__all__
    except ImportError:
        raise ImportError('Could not import strategies module. Ensure it is correctly defined.')

    for strategy_name in strategies:
        strategy_class = getattr(importlib.import_module('quanteval.strategies'), strategy_name)
        strategy_instance = strategy_class()
        signal = strategy_instance.generate_signals(sample_market_data)
        assert isinstance(signal, pd.Series), f'{strategy_name} did not return a pandas Series.'
        assert set(signal.dropna().unique()).issubset({0, 1}), (
            f'{strategy_name} returned non-binary signals.'
        )
