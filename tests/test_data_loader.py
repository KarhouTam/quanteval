from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from quanteval import DataLoader


def test_load_stock_uses_cache_when_available(
    tmp_path: Path, sample_market_data: pd.DataFrame
) -> None:
    cache_file = tmp_path / '600519_20200101_20231231_qfq.parquet'
    sample_market_data.to_parquet(cache_file)

    loader = DataLoader(cache_dir=str(tmp_path), use_cache=True)
    result = loader.load_stock('600519', '20200101', '20231231')

    pd.testing.assert_frame_equal(result, sample_market_data, check_freq=False)


def test_load_stock_standardizes_columns(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            '日期': ['2020-01-01', '2020-01-02', '2020-01-03'],
            '开盘': [10.0, 10.5, 11.0],
            '收盘': [10.2, 10.8, 10.9],
            '最高': [10.3, 10.9, 11.1],
            '最低': [9.9, 10.4, 10.7],
            '成交量': [100, 200, 150],
            '成交额': [1000, 2200, 1700],
        }
    )

    loader = DataLoader(cache_dir=str(tmp_path), use_cache=False)
    with patch('quanteval.data.loader.ak.stock_zh_a_hist', return_value=raw):
        result = loader.load_stock('600519', '20200101', '20200103')

    assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Ret']
    assert result.index.name == 'Date'


def test_load_hk_stock_standardizes_columns(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            '日期': ['2023-01-03', '2023-01-04', '2023-01-05'],
            '开盘': [368.0, 372.0, 375.0],
            '收盘': [371.0, 374.0, 370.0],
            '最高': [373.0, 376.0, 377.0],
            '最低': [366.0, 370.0, 368.0],
            '成交量': [10000, 12000, 9500],
            '成交额': [3710000.0, 4488000.0, 3515000.0],
        }
    )

    loader = DataLoader(cache_dir=str(tmp_path), use_cache=False)
    with patch('quanteval.data.loader.ak.stock_hk_hist', return_value=raw):
        result = loader.load_hk_stock('00700', '20230101', '20230131')

    assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Ret']
    assert result.index.name == 'Date'
    assert result.index.tz is None


def test_load_hk_stock_raises_on_invalid_symbol(tmp_path: Path) -> None:
    loader = DataLoader(cache_dir=str(tmp_path), use_cache=False)

    with pytest.raises(ValueError):
        loader.load_hk_stock('600519', '20230101', '20231231')

    with pytest.raises(ValueError):
        loader.load_hk_stock('ABC', '20230101', '20231231')


def test_load_hk_index_strips_timezone(tmp_path: Path) -> None:
    dates = pd.date_range('2023-01-03', periods=3, freq='B', tz='Asia/Hong_Kong')
    mock_df = pd.DataFrame(
        {
            'Open': [19500.0, 19600.0, 19450.0],
            'High': [19700.0, 19750.0, 19600.0],
            'Low': [19400.0, 19500.0, 19300.0],
            'Close': [19600.0, 19700.0, 19500.0],
            'Volume': [1000000, 1200000, 950000],
        },
        index=dates,
    )

    loader = DataLoader(cache_dir=str(tmp_path), use_cache=False)
    mock_ticker = patch('quanteval.data.loader.yf.Ticker')
    with mock_ticker as mock_cls:
        mock_cls.return_value.history.return_value = mock_df
        result = loader.load_hk_index('^HSI', '20230101', '20231231')

    assert result.index.tz is None
    assert result.index.name == 'Date'
    assert 'Ret' in result.columns
