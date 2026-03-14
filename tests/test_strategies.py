import pandas as pd

from quanteval import BollingerMeanReversionStrategy, BuyAndHoldStrategy, DualMAStrategy


def test_buy_and_hold_always_long(sample_market_data: pd.DataFrame) -> None:
    signal = BuyAndHoldStrategy().generate_signals(sample_market_data)
    assert signal.eq(1).all()


def test_dual_ma_returns_binary_signal(sample_market_data: pd.DataFrame) -> None:
    signal = DualMAStrategy(5, 20).generate_signals(sample_market_data)
    assert isinstance(signal, pd.Series)
    assert set(signal.dropna().unique()).issubset({0, 1})


def test_bollinger_strategy_returns_position_series(sample_market_data: pd.DataFrame) -> None:
    signal = BollingerMeanReversionStrategy(window=20, num_std=2.0).generate_signals(
        sample_market_data
    )
    assert signal.index.equals(sample_market_data.index)
    assert set(signal.unique()).issubset({0.0, 1.0})
