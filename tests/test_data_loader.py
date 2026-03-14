from pathlib import Path
from unittest.mock import patch

import pandas as pd

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
