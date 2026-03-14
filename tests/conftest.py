import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    index = pd.date_range('2022-01-03', periods=240, freq='B')
    base = np.linspace(100.0, 135.0, len(index))
    seasonality = np.sin(np.arange(len(index)) / 5) * 2
    close = pd.Series(base + seasonality, index=index)
    data = pd.DataFrame(
        {
            'Open': close * 0.99,
            'High': close * 1.01,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.linspace(1000, 2000, len(index)),
            'Amount': close * 1000,
        },
        index=index,
    )
    data['Ret'] = data['Close'].pct_change().fillna(0.0)
    return data


@pytest.fixture
def sample_benchmark_data(sample_market_data: pd.DataFrame) -> pd.DataFrame:
    benchmark = sample_market_data.copy()
    benchmark['Close'] = benchmark['Close'] * 0.995
    benchmark['Ret'] = benchmark['Close'].pct_change().fillna(0.0)
    return benchmark
