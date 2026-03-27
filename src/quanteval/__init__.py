"""QuantEval public package surface.

Expose a minimal, stable API for most users. Keep surface area small to
avoid importing heavy internals on package import.
"""

__version__ = '1.0.0'

# Core
from quanteval.core.strategy import Strategy
from quanteval.core.backtester import Backtester, BacktestResults
from quanteval.core.transaction import TransactionCost, HKTransactionCost

# Data
from quanteval.data.loader import DataLoader

# Metrics
from quanteval.metrics.performance import PerformanceMetrics

# Factor base
from quanteval.factors.base import Factor

# Small set of utility helpers
from quanteval.utils.helpers import (
    calculate_returns,
    calculate_cumulative_returns,
    configure_chinese_font,
)

__all__ = [
    # Core
    'Strategy',
    'Backtester',
    'BacktestResults',
    # Transaction Costs
    'TransactionCost',
    'HKTransactionCost',
    # Data
    'DataLoader',
    # Metrics
    'PerformanceMetrics',
    # Factors
    'Factor',
    # Utilities
    'calculate_returns',
    'align_series',
    'calculate_cumulative_returns',
]

configure_chinese_font()
