"""Strategies module - Benchmark trading strategies."""

from quanteval.strategies.dual_ma import DualMAStrategy
from quanteval.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from quanteval.strategies.buy_hold import BuyAndHoldStrategy
from quanteval.strategies.dual_thrust import DualThrustStrategy
from quanteval.strategies.rsi_reversion import RSIStrategy
from quanteval.strategies.donchain_channel import DonchianChannel

__all__ = [
    'DualMAStrategy',
    'BollingerMeanReversionStrategy',
    'BuyAndHoldStrategy',
    'DualThrustStrategy',
    'RSIStrategy',
    'DonchianChannel',
]
