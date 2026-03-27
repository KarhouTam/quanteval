"""Aggregators module initialization."""

from quanteval.aggregators.base import FactorAggregator
from quanteval.aggregators.weighted import EqualWeightAggregator, ICWeightedAggregator

__all__ = ['FactorAggregator', 'EqualWeightAggregator', 'ICWeightedAggregator']
