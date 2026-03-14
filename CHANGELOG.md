# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-03-14

### Added
- Backtesting engine with delayed signal execution to reduce look-ahead bias
- Built-in strategies: dual moving average cross, Bollinger band mean reversion, buy-and-hold benchmark
- Technical factor library: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Momentum, VolumeMA, ROC, Stochastic Oscillator
- Grid search and walk-forward parameter optimisation
- Multi-strategy comparison framework with correlation analysis and equal-weight portfolio construction
- Static (matplotlib) and interactive (Plotly) visualisation for equity curves and drawdowns
- AkShare data loader with local Parquet caching for Chinese A-share, ETF, and index data
- GitHub Actions CI: lint, pytest (Python 3.11–3.13), notebook smoke tests, build validation
- GitHub Actions release workflow: tag-triggered artifact build and GitHub Release creation
- Bilingual documentation: English README and Chinese README, contributing guide, and tutorial
- Example Jupyter notebooks: quickstart, parameter optimisation, strategy comparison

### Notes
- Requires Python >= 3.11

[0.1.0]: https://github.com/KarhouTam/quanteval/releases/tag/v0.1.0
