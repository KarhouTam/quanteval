"""Utilities module - Helper functions."""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    计算收益率 (Calculate returns)

    Args:
        prices: Price series

    Returns:
        Daily returns series
    """
    return prices.pct_change()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    计算累计收益 (Calculate cumulative returns)

    Args:
        returns: Daily returns series

    Returns:
        Cumulative returns series
    """
    return (1 + returns).cumprod()


def align_series(series1: pd.Series, series2: pd.Series) -> tuple:
    """
    对齐两个时间序列 (Align two time series)

    Args:
        series1: First series
        series2: Second series

    Returns:
        Tuple of aligned series
    """
    common_index = series1.index.intersection(series2.index)
    return series1.loc[common_index], series2.loc[common_index]


def configure_chinese_font() -> None:
    """
    配置 Matplotlib 中文字体显示。

    自动检测系统中常见中文字体并设置，若未找到则保持默认字体。
    同时关闭负号乱码问题。
    """
    preferred_fonts = [
        'Microsoft YaHei',
        'SimHei',
        'PingFang SC',
        'Hiragino Sans GB',
        'Heiti SC',
        'Noto Sans CJK SC',
        'WenQuanYi Zen Hei',
        'Arial Unicode MS',
        'STSong',
    ]

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    selected_font = next((name for name in preferred_fonts if name in available_fonts), None)

    if selected_font is not None:
        plt.rcParams['font.sans-serif'] = [selected_font] + list(
            plt.rcParams.get('font.sans-serif', [])
        )

    plt.rcParams['axes.unicode_minus'] = False
