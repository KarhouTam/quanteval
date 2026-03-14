"""Visualization module - Interactive plotly charts."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_backtest_results(results):
    """
    绘制交互式回测结果 (Plot interactive backtest results)

    Args:
        results: BacktestResults instance

    Returns:
        plotly Figure object
    """
    # Calculate drawdown
    peak = results.equity_curve.cummax()
    drawdown = (results.equity_curve - peak) / peak

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f'{results.strategy_name} - 收益曲线（Equity）',
            '回撤（Drawdown）',
            '仓位（Position）',
        ),
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=results.equity_curve.index,
            y=results.equity_curve.values,
            name=results.strategy_name,
            line=dict(color='red', width=2),
        ),
        row=1,
        col=1,
    )

    if results.benchmark_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=results.benchmark_equity.index,
                y=results.benchmark_equity.values,
                name='基准 (Buy & Hold)',
                line=dict(color='gray', width=1.5),
                opacity=0.6,
            ),
            row=1,
            col=1,
        )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='回撤（Drawdown）',
            fill='tozeroy',
            line=dict(color='red', width=1),
            fillcolor='rgba(255,0,0,0.3)',
        ),
        row=2,
        col=1,
    )

    # Position
    fig.add_trace(
        go.Scatter(
            x=results.positions.index,
            y=results.positions.values,
            name='仓位（Position）',
            fill='tozeroy',
            line=dict(color='green', width=1),
            fillcolor='rgba(0,255,0,0.3)',
        ),
        row=3,
        col=1,
    )

    # Update layout
    fig.update_layout(height=800, hovermode='x unified', showlegend=True, template='plotly_white')

    fig.update_yaxes(title_text='收益曲线（Equity）', row=1, col=1)
    fig.update_yaxes(title_text='回撤（Drawdown）', row=2, col=1)
    fig.update_yaxes(title_text='仓位（Position）', row=3, col=1)
    fig.update_xaxes(title_text='日期（Date）', row=3, col=1)

    return fig
