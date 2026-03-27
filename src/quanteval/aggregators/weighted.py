"""
Factor Aggregator - 因子聚合器
Combine multiple factors into composite signals with various weighting schemes.
"""

import numpy as np
import pandas as pd

from quanteval.aggregators.base import FactorAggregator


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    滚动 Z-Score 标准化

    Rolling z-score normalization. Guards against std=0.

    Args:
        series: Input series to normalize
        window: Rolling window size
        min_periods: Minimum number of observations required

    Returns:
        Normalized series with same index
    """
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    std = std.replace(0, np.nan)  # Prevent division by zero
    return (series - mean) / std


def _rolling_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    滚动百分位排名

    Rolling percentile rank.

    Args:
        series: Input series to rank
        window: Rolling window size
        min_periods: Minimum number of observations required

    Returns:
        Ranked series (0-1 scale) with same index
    """
    return series.rolling(window=window, min_periods=min_periods).rank(pct=True)


def _rolling_winsorize(
    series: pd.Series, window: int, n_sigma: float, min_periods: int
) -> pd.Series:
    """
    滚动缩尾处理

    Clip values beyond mean ± n_sigma*std per rolling window.

    Args:
        series: Input series to winsorize
        window: Rolling window size
        n_sigma: Number of standard deviations for clipping
        min_periods: Minimum number of observations required

    Returns:
        Winsorized series with same index
    """
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    lower = mean - n_sigma * std
    upper = mean + n_sigma * std
    return series.clip(lower=lower, upper=upper, axis=0)


def _rolling_spearman_corr(
    series1: pd.Series, series2: pd.Series, window: int, min_periods: int
) -> pd.Series:
    """
    滚动 Spearman 相关系数

    Rolling Spearman correlation coefficient between two series.

    Args:
        series1: First series
        series2: Second series
        window: Rolling window size
        min_periods: Minimum number of observations required

    Returns:
        Rolling Spearman correlation series with same index as series1

    Notes:
        Uses rank transformation followed by Pearson correlation on ranks.
    """

    def spearman_corr(x, y):
        """Compute Spearman correlation between two arrays."""
        if len(x) < 2 or len(y) < 2:
            return np.nan
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) < 2:
            return np.nan
        rank_x = pd.Series(x_clean).rank()
        rank_y = pd.Series(y_clean).rank()
        return rank_x.corr(rank_y)

    aligned = pd.DataFrame({'s1': series1, 's2': series2})

    result = []
    for i in range(len(aligned)):
        if i < min_periods - 1:
            result.append(np.nan)
        else:
            start_idx = max(0, i - window + 1)
            window_data = aligned.iloc[start_idx : i + 1]
            corr = spearman_corr(window_data['s1'].values, window_data['s2'].values)
            result.append(corr)

    return pd.Series(result, index=series1.index)


class EqualWeightAggregator(FactorAggregator):
    """
    等权重因子聚合器

    Equal-weight factor aggregation with optional rolling z-score normalization.

    Args:
        normalize: Whether to apply rolling z-score normalization (default: True)
        normalize_window: Rolling window for normalization (default: 60)
        min_periods: Minimum observations required for normalization (default: 20)

    Example:
        >>> aggregator = EqualWeightAggregator(normalize=True, normalize_window=60)
        >>> composite = aggregator.aggregate(factors_dict, data)
    """

    def __init__(
        self,
        normalize: bool = True,
        normalize_window: int = 60,
        min_periods: int = 20,
    ):
        super().__init__(name='EqualWeightAggregator')
        self.normalize = normalize
        self.normalize_window = normalize_window
        self.min_periods = min_periods

    def aggregate(self, factors: dict[str, pd.Series], data: pd.DataFrame) -> pd.Series:
        """
        聚合因子 (等权重)

        Aggregate factors using equal weighting. Optionally applies rolling
        z-score normalization before averaging.

        Args:
            factors: Dict mapping factor names to their Series values
            data: Original OHLCV DataFrame (not used in equal-weight scheme)

        Returns:
            Composite score Series with same index as data
        """
        if not factors:
            return pd.Series(np.nan, index=data.index, name='CompositeScore')

        aligned_factors = []
        for factor_series in factors.values():
            aligned = factor_series.reindex(data.index)
            if self.normalize:
                aligned = _rolling_zscore(aligned, self.normalize_window, self.min_periods)
            aligned_factors.append(aligned)

        composite = pd.concat(aligned_factors, axis=1).mean(axis=1)
        composite.name = 'CompositeScore'
        return composite


class ICWeightedAggregator(FactorAggregator):
    """
    IC 加权因子聚合器

    Information Coefficient (IC) weighted factor aggregation. Computes rolling
    Spearman correlation between each factor and forward returns, then uses
    shifted IC as weights to prevent look-ahead bias.

    Args:
        ic_window: Rolling window for IC calculation (default: 60)
        min_periods: Minimum observations required for IC calculation (default: 20)
        normalize: Whether to apply rolling z-score normalization (default: True)
        normalize_window: Rolling window for normalization (default: 60)

    Example:
        >>> aggregator = ICWeightedAggregator(ic_window=60, normalize=True)
        >>> composite = aggregator.aggregate(factors_dict, data)

    Notes:
        - Forward returns are computed as Close.pct_change().shift(-1)
        - IC weights are shifted by 1 period to prevent look-ahead bias
        - Falls back to equal weights when IC is all NaN or sums to zero
    """

    def __init__(
        self,
        ic_window: int = 60,
        min_periods: int = 20,
        normalize: bool = True,
        normalize_window: int = 60,
    ):
        super().__init__(name='ICWeightedAggregator')
        self.ic_window = ic_window
        self.min_periods = min_periods
        self.normalize = normalize
        self.normalize_window = normalize_window

    def aggregate(self, factors: dict[str, pd.Series], data: pd.DataFrame) -> pd.Series:
        """
        聚合因子 (IC 加权)

        Aggregate factors using rolling IC weights. Falls back to equal weights
        when IC is unavailable.

        Args:
            factors: Dict mapping factor names to their Series values
            data: Original OHLCV DataFrame (must contain 'Close' column)

        Returns:
            Composite score Series with same index as data
        """
        if not factors:
            return pd.Series(np.nan, index=data.index, name='CompositeScore')

        forward_returns = data['Close'].pct_change().shift(-1)

        aligned_factors = []
        ic_weights_list = []

        for factor_series in factors.values():
            aligned = factor_series.reindex(data.index)

            rolling_ic = _rolling_spearman_corr(
                aligned, forward_returns, self.ic_window, self.min_periods
            )

            ic_weight = rolling_ic.shift(1)
            ic_weights_list.append(ic_weight)

            if self.normalize:
                aligned = _rolling_zscore(aligned, self.normalize_window, self.min_periods)
            aligned_factors.append(aligned)

        ic_weights_df = pd.concat(ic_weights_list, axis=1)
        ic_sum = ic_weights_df.sum(axis=1)
        normalized_weights = ic_weights_df.div(ic_sum, axis=0)

        equal_weight = 1.0 / len(factors)
        normalized_weights = normalized_weights.fillna(equal_weight)

        factors_df = pd.concat(aligned_factors, axis=1)
        weighted_factors = factors_df * normalized_weights.values

        composite = weighted_factors.sum(axis=1)
        composite.name = 'CompositeScore'
        return composite
