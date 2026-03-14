import pandas as pd

from quanteval import (
    ATR,
    BollingerBands,
    MACD,
    Momentum,
    ROC,
    RSI,
    SMA,
    StochasticOscillator,
    VolumeMA,
)


def test_single_series_factors_return_series(sample_market_data: pd.DataFrame) -> None:
    factors = [SMA(5), RSI(14), ATR(14), Momentum(10), VolumeMA(5), ROC(5)]
    for factor in factors:
        result = factor.calculate(sample_market_data)
        assert isinstance(result, pd.Series)
        assert result.index.equals(sample_market_data.index)


def test_macd_and_bollinger_return_expected_keys(sample_market_data: pd.DataFrame) -> None:
    macd = MACD().calculate(sample_market_data)
    bands = BollingerBands().calculate(sample_market_data)
    stoch = StochasticOscillator().calculate(sample_market_data)

    assert set(macd) == {'macd', 'signal', 'histogram'}
    assert set(bands) == {'middle', 'upper', 'lower'}
    assert set(stoch) == {'k', 'd'}
