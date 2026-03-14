"""
QuantEval - 量化策略与因子评估平台
A comprehensive platform for evaluating quantitative trading strategies and factors.

Author: KarhouTam
License: MIT
"""

__version__ = '1.0.0'

# Core components
from quanteval.core.strategy import Strategy
from quanteval.core.backtester import (
    Backtester,
    BacktestResults,
    MultiBacktestResults,
    MetricsSnapshot,
)
from quanteval.core.transaction import TransactionCost, HKTransactionCost

# Data management
from quanteval.data.loader import DataLoader

# Metrics
from quanteval.metrics.performance import PerformanceMetrics

# Factors
from quanteval.factors.base import Factor
from quanteval.factors.technical import (
    SMA,
    EMA,
    RSI,
    MACD,
    BollingerBands,
    ATR,
    Momentum,
    VolumeMA,
    ROC,
    StochasticOscillator,
)

# Benchmark strategies
from quanteval.strategies.dual_ma import DualMAStrategy
from quanteval.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from quanteval.strategies.buy_hold import BuyAndHoldStrategy

# Optimization
from quanteval.optimization import GridSearch, WalkForwardAnalysis

# Comparison
from quanteval.comparison import StrategyComparator, StrategyComparison
from quanteval.utils.helpers import (
    calculate_returns,
    calculate_cumulative_returns,
    align_series,
    configure_chinese_font,
)

__all__ = [
    # Core
    'Strategy',
    'Backtester',
    'BacktestResults',
    'MultiBacktestResults',
    'MetricsSnapshot',
    'TransactionCost',
    'HKTransactionCost',
    # Data
    'DataLoader',
    # Metrics
    'PerformanceMetrics',
    # Factors
    'Factor',
    'SMA',
    'EMA',
    'RSI',
    'MACD',
    'BollingerBands',
    'ATR',
    'Momentum',
    'VolumeMA',
    'ROC',
    'StochasticOscillator',
    # Strategies
    'DualMAStrategy',
    'BollingerMeanReversionStrategy',
    'BuyAndHoldStrategy',
    # Optimization
    'GridSearch',
    'WalkForwardAnalysis',
    # Comparison
    'StrategyComparator',
    'StrategyComparison',
    'calculate_returns',
    'calculate_cumulative_returns',
    'align_series',
    'configure_chinese_font',
]
