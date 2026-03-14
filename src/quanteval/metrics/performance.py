"""
Performance Metrics - 绩效指标计算
Comprehensive suite of performance metrics for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class PerformanceMetrics:
    """
    绩效指标计算器 - 计算策略的各类绩效指标

    Performance metrics calculator for trading strategies.

    Args:
        strategy_returns: 策略日收益率序列 (Strategy daily returns Series)
        benchmark_returns: 基准日收益率序列 (Benchmark daily returns Series, optional)
        positions: 仓位序列 (Position Series, optional, for trade statistics)
        risk_free_rate: 无风险利率年化 (Annual risk-free rate, default 0.03)
        trading_days: 每年交易日数 (Trading days per year, default 252)
    """

    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        positions: Optional[pd.Series] = None,
        risk_free_rate: float = 0.03,
        trading_days: int = 252,
    ):
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns
        self.positions = positions
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

        # Calculate equity curves
        self.strategy_equity = (1 + strategy_returns).cumprod()
        if benchmark_returns is not None:
            self.benchmark_equity = (1 + benchmark_returns).cumprod()
        else:
            self.benchmark_equity = None

    def total_return(self) -> float:
        """总收益率 (Total return)"""
        return self.strategy_equity.iloc[-1] - 1

    def annual_return(self) -> float:
        """年化收益率 (Annualized return)"""
        total_days = len(self.strategy_returns)
        total_ret = self.total_return()
        years = total_days / self.trading_days
        return (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    def annual_volatility(self) -> float:
        """年化波动率 (Annualized volatility)"""
        return self.strategy_returns.std() * np.sqrt(self.trading_days)

    def sharpe_ratio(self) -> float:
        """
        夏普比率 (Sharpe ratio)

        Formula: (Annual Return - Risk Free Rate) / Annual Volatility
        """
        annual_ret = self.annual_return()
        annual_vol = self.annual_volatility()

        if annual_vol == 0:
            return 0.0

        return (annual_ret - self.risk_free_rate) / annual_vol

    def sortino_ratio(self) -> float:
        """
        索提诺比率 (Sortino ratio)

        Only penalizes downside volatility.
        """
        annual_ret = self.annual_return()

        # Calculate downside deviation
        downside_returns = self.strategy_returns[self.strategy_returns < 0]

        if len(downside_returns) == 0:
            return np.inf

        downside_std = downside_returns.std() * np.sqrt(self.trading_days)

        if downside_std == 0:
            return 0.0

        return (annual_ret - self.risk_free_rate) / downside_std

    def max_drawdown(self) -> float:
        """
        最大回撤 (Maximum drawdown)

        Returns:
            Maximum drawdown as negative value (e.g., -0.18 for 18% drawdown)
        """
        peak = self.strategy_equity.cummax()
        drawdown = (self.strategy_equity - peak) / peak
        return drawdown.min()

    def max_drawdown_duration(self) -> int:
        """
        最大回撤持续期 (Maximum drawdown duration in days)
        """
        peak = self.strategy_equity.cummax()
        drawdown = (self.strategy_equity - peak) / peak

        # Find periods where drawdown exists
        is_drawdown = drawdown < 0

        if not is_drawdown.any():
            return 0

        # Calculate duration of each drawdown period
        drawdown_periods = []
        current_duration = 0

        for dd in is_drawdown:
            if dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            drawdown_periods.append(current_duration)

        return max(drawdown_periods) if drawdown_periods else 0

    def calmar_ratio(self) -> float:
        """
        卡玛比率 (Calmar ratio)

        Formula: Annual Return / |Maximum Drawdown|
        """
        annual_ret = self.annual_return()
        max_dd = abs(self.max_drawdown())

        if max_dd == 0:
            return 0.0

        return annual_ret / max_dd

    def win_rate(self) -> float:
        """
        胜率 (Win rate) - requires positions

        Returns:
            Percentage of winning trades
        """
        if self.positions is None:
            return np.nan

        trades = self._extract_trades()

        if len(trades) == 0:
            return np.nan

        winning_trades = sum(1 for t in trades if t > 0)
        return winning_trades / len(trades)

    def profit_loss_ratio(self) -> float:
        """
        盈亏比 (Profit/Loss ratio) - requires positions

        Returns:
            Average win / |Average loss|
        """
        if self.positions is None:
            return np.nan

        trades = self._extract_trades()

        if len(trades) == 0:
            return np.nan

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]

        if len(wins) == 0 or len(losses) == 0:
            return np.nan

        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return np.inf

        return avg_win / avg_loss

    def num_trades(self) -> int:
        """交易次数 (Number of trades) - requires positions"""
        if self.positions is None:
            return 0

        return len(self._extract_trades())

    def avg_trade_duration(self) -> float:
        """平均持仓天数 (Average trade duration in days) - requires positions"""
        if self.positions is None:
            return np.nan

        trade_durations = self._extract_trade_durations()

        if len(trade_durations) == 0:
            return np.nan

        return np.mean(trade_durations)

    def alpha(self) -> float:
        """
        Alpha - 相对基准的超额收益 (Excess return vs benchmark)

        Simple alpha: Strategy Annual Return - Benchmark Annual Return
        """
        if self.benchmark_returns is None:
            return np.nan

        strategy_annual = self.annual_return()

        total_days = len(self.benchmark_returns)
        benchmark_total = self.benchmark_equity.iloc[-1] - 1
        years = total_days / self.trading_days
        benchmark_annual = (1 + benchmark_total) ** (1 / years) - 1 if years > 0 else 0

        return strategy_annual - benchmark_annual

    def information_ratio(self) -> float:
        """
        信息比率 (Information ratio)

        Formula: Alpha / Tracking Error
        """
        if self.benchmark_returns is None:
            return np.nan

        # Active returns
        active_returns = self.strategy_returns - self.benchmark_returns

        # Tracking error (annualized)
        tracking_error = active_returns.std() * np.sqrt(self.trading_days)

        if tracking_error == 0:
            return 0.0

        # Annualized alpha
        alpha = active_returns.mean() * self.trading_days

        return alpha / tracking_error

    def _extract_trades(self) -> list:
        """
        提取所有交易的盈亏

        Extract profit/loss for each trade.

        Returns:
            List of trade returns (as decimals, e.g., 0.05 for 5% gain)
        """
        if self.positions is None:
            return []

        # Create a copy to avoid modifying original
        df = pd.DataFrame({'Position': self.positions, 'Ret': self.strategy_returns})

        # Mark trade changes
        df['Trade_Mark'] = df['Position'].diff()

        # Entry points (0 -> 1)
        entry_dates = df[df['Trade_Mark'] == 1].index

        # Exit points (1 -> 0)
        exit_dates = df[df['Trade_Mark'] == -1].index

        # Match entries with exits
        trades = []
        for entry in entry_dates:
            # Find corresponding exit
            exits_after_entry = exit_dates[exit_dates > entry]

            if len(exits_after_entry) > 0:
                exit = exits_after_entry[0]

                # Calculate trade return
                trade_data = df.loc[entry:exit, 'Ret']
                trade_return = (1 + trade_data).prod() - 1
                trades.append(trade_return)

        return trades

    def _extract_trade_durations(self) -> list:
        """
        提取所有交易的持续期

        Extract duration for each trade.

        Returns:
            List of trade durations in days
        """
        if self.positions is None:
            return []

        df = pd.DataFrame({'Position': self.positions})
        df['Trade_Mark'] = df['Position'].diff()

        entry_dates = df[df['Trade_Mark'] == 1].index
        exit_dates = df[df['Trade_Mark'] == -1].index

        durations = []
        for entry in entry_dates:
            exits_after_entry = exit_dates[exit_dates > entry]

            if len(exits_after_entry) > 0:
                exit = exits_after_entry[0]
                duration = (exit - entry).days
                durations.append(duration)

        return durations

    def calculate_all(self) -> Dict[str, float]:
        """
        计算所有指标

        Calculate all metrics and return as dictionary.

        Returns:
            Dictionary with all performance metrics
        """
        metrics = {
            # Returns
            'total_return': self.total_return(),
            'annual_return': self.annual_return(),
            # Risk
            'annual_volatility': self.annual_volatility(),
            'max_drawdown': self.max_drawdown(),
            'max_drawdown_duration': self.max_drawdown_duration(),
            # Risk-adjusted
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            # Trading statistics
            'num_trades': self.num_trades(),
            'win_rate': self.win_rate(),
            'profit_loss_ratio': self.profit_loss_ratio(),
            'avg_trade_duration': self.avg_trade_duration(),
        }

        # Add benchmark comparison metrics if available
        if self.benchmark_returns is not None:
            metrics['benchmark_return'] = (
                self.benchmark_equity.iloc[-1] - 1 if self.benchmark_equity is not None else np.nan
            )
            metrics['alpha'] = self.alpha()
            metrics['information_ratio'] = self.information_ratio()

        return metrics

    def summary(self) -> pd.Series:
        """
        返回格式化的指标摘要

        Return formatted metrics summary as Series.
        """
        metrics = self.calculate_all()
        return pd.Series(metrics)
