"""
Data Loader - AkShare数据接口封装
Wrapper for AkShare API with caching and standardized DataFrame format.
"""

import akshare as ak
import pandas as pd
from pathlib import Path
from typing import Optional
import warnings
import re
import yfinance as yf


class DataLoader:
    """
    数据加载器 - 支持股票和指数数据获取，带本地缓存功能

    Data loader with caching support for Chinese A-share stocks and indices.

    Args:
        cache_dir: 缓存目录路径 (Cache directory path)
        use_cache: 是否使用缓存 (Whether to use cache)
    """

    def __init__(self, cache_dir: str = './data_cache', use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_stock(
        self, symbol: str, start_date: str, end_date: Optional[str] = None, adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        加载A股股票历史数据

        Load historical data for A-share stocks.

        Args:
            symbol: 股票代码，6位数字 (Stock code, 6 digits, e.g., '600519')
            start_date: 开始日期 'YYYYMMDD' (Start date)
            end_date: 结束日期 'YYYYMMDD'，默认今天 (End date, default today)
            adjust: 复权类型 'qfq'前复权/'hfq'后复权/''不复权 (Adjustment type)

        Returns:
            DataFrame with DatetimeIndex and columns:
            ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Ret']
        """
        resolved_end_date = end_date or pd.Timestamp.today().strftime('%Y%m%d')

        # Generate cache filename
        cache_file = self.cache_dir / f'{symbol}_{start_date}_{resolved_end_date}_{adjust}.parquet'

        # Try to load from cache
        if self.use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
            return df

        # Fetch from AkShare
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol, start_date=start_date, end_date=resolved_end_date, adjust=adjust
            )
        except Exception as e:
            raise ValueError(f'Failed to fetch data for {symbol}: {e}') from e

        if df.empty:
            raise ValueError(
                f'No data available for {symbol} from {start_date} to {resolved_end_date}'
            )

        # Standardize column names (AkShare returns Chinese columns)
        column_mapping = {
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume',
            '成交额': 'Amount',
            '振幅': 'Amplitude',
            '涨跌幅': 'Pct_Chg',
            '涨跌额': 'Change',
            '换手率': 'Turnover',
        }

        df = df.rename(columns=column_mapping)

        # Set Date as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        # Keep only essential columns (OHLCV + Amount)
        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols]

        # Calculate daily returns
        close_series = pd.Series(df['Close'], index=df.index, dtype=float)
        df['Ret'] = close_series.pct_change()

        # Remove rows with NaN
        df = df.dropna()

        # Save to cache
        if self.use_cache:
            df.to_parquet(cache_file)

        return df

    def load_index(
        self, symbol: str, start_date: str, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载指数历史数据

        Load historical data for indices.

        Args:
            symbol: 指数代码 (Index code, e.g., '000300' for CSI 300)
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'

        Returns:
            DataFrame with same format as load_stock
        """
        # Generate cache filename
        cache_file = self.cache_dir / f'index_{symbol}_{start_date}_{end_date}.parquet'

        # Try to load from cache
        if self.use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
            return df

        # Fetch from AkShare
        try:
            df = ak.stock_zh_index_daily(symbol=f'sh{symbol}')
        except Exception as e:
            warnings.warn(
                f"Failed to fetch index {symbol} with 'sh' prefix, trying 'sz': {e}",
                stacklevel=2,
            )
            try:
                df = ak.stock_zh_index_daily(symbol=f'sz{symbol}')
            except Exception as e2:
                raise ValueError(f'Failed to fetch index data for {symbol}: {e2}') from e2

        if df.empty:
            raise ValueError(f'No data available for index {symbol}')

        # Standardize columns
        df = df.rename(
            columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            }
        )

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        # Filter date range
        df = df.loc[start_date : end_date if end_date else df.index[-1]]

        # Calculate returns
        df['Ret'] = df['Close'].pct_change()
        df = df.dropna()

        # Save to cache
        if self.use_cache:
            df.to_parquet(cache_file)

        return df

    def load_etf(
        self, symbol: str, start_date: str, end_date: Optional[str] = None, adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        加载ETF历史数据

        Load historical data for ETFs.

        Args:
            symbol: ETF代码 (ETF code, e.g., '510300')
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
            adjust: 复权类型 'qfq'前复权/'hfq'后复权/''不复权

        Returns:
            DataFrame with same format as load_stock
        """
        cache_file = self.cache_dir / f'etf_{symbol}_{start_date}_{end_date}_{adjust}.parquet'

        if self.use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
            return df

        try:
            # Sina requires sh/sz prefix
            market_prefix = 'sh' if symbol.startswith('5') else 'sz'
            full_symbol = f'{market_prefix}{symbol}'

            df = ak.fund_etf_hist_sina(symbol=full_symbol)

            # Filter date range
            df['date'] = pd.to_datetime(df['date'])
            mask = df['date'] >= pd.to_datetime(start_date)
            if end_date:
                mask &= df['date'] <= pd.to_datetime(end_date)
            df = df.loc[mask].copy()

        except Exception as e:
            raise ValueError(f'Failed to fetch ETF data for {symbol}: {e}') from e

        if df.empty:
            raise ValueError(f'No data available for ETF {symbol}')

        column_mapping = {
            'date': 'Date',
            'open': 'Open',
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume',
            'amount': 'Amount',
        }

        df = df.rename(columns=column_mapping)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols]

        df['Ret'] = df['Close'].pct_change()
        df = df.dropna()

        if self.use_cache:
            df.to_parquet(cache_file)

        return df

    def load_hk_stock(
        self, symbol: str, start_date: str, end_date: Optional[str] = None, adjust: str = ''
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data for Hong Kong (HKEX) stocks via AKShare.

        Args:
            symbol: HKEX stock code, 5-digit zero-padded string (e.g., '00700' for Tencent)
            start_date: Start date 'YYYYMMDD'
            end_date: End date 'YYYYMMDD', default today
            adjust: Price adjustment type: '' no adjustment, 'qfq' forward, 'hfq' backward

        Returns:
            DataFrame with DatetimeIndex (timezone-naive, named 'Date') and columns:
            ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Ret']

        Raises:
            ValueError: If symbol is not a 5-digit string, adjust is invalid,
                        data fetch fails, or no data is available.
        """
        if not re.match(r'^\d{5}$', symbol):
            raise ValueError(
                f"Invalid HK stock symbol '{symbol}': must be a 5-digit zero-padded string (e.g., '00700')"
            )
        valid_adjust = ('', 'qfq', 'hfq')
        if adjust not in valid_adjust:
            raise ValueError(f"Invalid adjust value '{adjust}': must be one of {valid_adjust}")

        resolved_end_date = end_date or pd.Timestamp.today().strftime('%Y%m%d')

        cache_file = (
            self.cache_dir / f'hk_{symbol}_{start_date}_{resolved_end_date}_{adjust}.parquet'
        )

        if self.use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)

        try:
            df = ak.stock_hk_hist(
                symbol=symbol,
                period='daily',
                start_date=start_date,
                end_date=resolved_end_date,
                adjust=adjust,
            )
        except Exception as e:
            raise ValueError(f'Failed to fetch HK data for {symbol}: {e}') from e

        if df.empty:
            raise ValueError(
                f'No data available for HK stock {symbol} from {start_date} to {resolved_end_date}'
            )

        column_mapping = {
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume',
            '成交额': 'Amount',
            '振幅': 'Amplitude',
            '涨跌幅': 'Pct_Chg',
            '涨跌额': 'Change',
            '换手率': 'Turnover',
        }
        df = df.rename(columns=column_mapping)

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols]

        df['Ret'] = pd.Series(df['Close'], index=df.index, dtype=float).pct_change()
        df = df.dropna()

        if self.use_cache:
            df.to_parquet(cache_file)

        return df

    def load_hk_index(
        self, symbol: str, start_date: str, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data for Hong Kong market indices via yfinance.

        Supported symbols: '^HSI' (Hang Seng Index), '^HSCE' (Hang Seng China Enterprises).
        Note: '^HSTECH' is NOT supported due to a known yfinance bug.

        Args:
            symbol: Index ticker symbol (e.g., '^HSI', '^HSCE')
            start_date: Start date 'YYYYMMDD'
            end_date: End date 'YYYYMMDD', default None (fetch up to today)

        Returns:
            DataFrame with DatetimeIndex (timezone-naive, named 'Date') and columns:
            ['Open', 'High', 'Low', 'Close', 'Volume', 'Ret']

        Raises:
            ValueError: If symbol is '^HSTECH', data fetch fails, or no data available.
        """
        if symbol == '^HSTECH':
            raise ValueError(
                "'^HSTECH' is not supported: yfinance has a known bug with this ticker "
                "(AttributeError on tzinfo). Use '3033.HK' ETF as a proxy if needed."
            )

        cache_sym = symbol.replace('^', '_')
        cache_file = self.cache_dir / f'hk_index_{cache_sym}_{start_date}_{end_date}.parquet'

        if self.use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)

        start_dt = pd.to_datetime(start_date, format='%Y%m%d')
        end_dt = pd.to_datetime(end_date, format='%Y%m%d') if end_date else None

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_dt, end=end_dt)
        except Exception as e:
            raise ValueError(f'Failed to fetch HK index data for {symbol}: {e}') from e

        if df.empty:
            raise ValueError(f'No data available for HK index {symbol}')

        # Strip timezone from yfinance DatetimeIndex (returns Asia/Hong_Kong tz-aware)
        df.index = df.index.tz_localize(None)
        df.index.name = 'Date'

        available_cols = [
            col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns
        ]
        df = df[available_cols].copy()

        df['Ret'] = df['Close'].pct_change()
        df = df.dropna()

        df = df.loc[start_dt:]

        if self.use_cache:
            df.to_parquet(cache_file)

        return df

    def clear_cache(self):
        """清空缓存目录 (Clear cache directory)"""
        if self.cache_dir.exists():
            for file in self.cache_dir.glob('*.parquet'):
                file.unlink()
            print(f'Cache cleared: {self.cache_dir}')
