# Weather Prediction Market Trading Simulator

## Overview

This project is a Python-based prediction market trading framework focused on weather events. The system converts weather forecasts into probabilistic estimates, transforms those probabilities into prediction-market fair values, identifies market mispricings, generates trading signals, manages inventory and risk, and evaluates performance through Monte Carlo simulation.

The primary contract studied throughout development was:

> Will Philadelphia receive more than 0.50 inches of rain tomorrow?

Although the project began with a single contract, it was later extended to support multiple cities and live weather forecast data.

---

## Motivation

Prediction markets can be viewed as probability markets where contract prices represent beliefs about future events.

For a binary YES contract:

* YES pays $1 if the event occurs
* YES pays $0 otherwise

If an event has a 70% probability of occurring, the fair value of a YES contract is approximately:

Fair Value = 0.70

The central goal of the project is to identify situations where market prices differ from forecast-implied fair values and trade accordingly.

---

## Core Features

### Contract Modeling

* Binary YES/NO weather contracts
* Event resolution logic
* Contract settlement and payout calculations

### Forecast Modeling

* Forecast-implied probability estimation
* Fair value calculation
* Edge calculation versus market prices

### Trading Engine

* BUY YES signals
* SELL YES signals
* NO TRADE decisions
* Dynamic position sizing

### Portfolio Management

* Cash tracking
* Inventory tracking
* Portfolio valuation
* Realized and unrealized PnL

### Risk Management

* Inventory limits
* Conservative trading mode
* Aggressive trading mode
* Trade blocking when limits are exceeded

### Bayesian Updating

* Sequential probability updates
* Bullish weather signals
* Bearish weather signals
* Neutral weather signals

### Backtesting

* Monte Carlo simulation framework
* Performance evaluation across many market paths
* Win-rate analysis
* Risk-adjusted performance metrics

### Visualization

* Market price evolution
* Model probability evolution
* Inventory history
* Portfolio value history
* PnL history
* Monte Carlo PnL distributions

### Live Data Integration

* Open-Meteo API integration
* Live precipitation forecasts
* Real-time probability estimation
* Multi-city forecast analysis

---

## Project Architecture

```text
Weather Forecast
        ↓
Probability Estimate
        ↓
Fair Value Calculation
        ↓
Market Price Comparison
        ↓
Edge Detection
        ↓
Trading Signal
        ↓
Position Sizing
        ↓
Risk Checks
        ↓
Trade Recommendation
        ↓
PnL Tracking
```

---

## Bayesian Forecast Updating

The project incorporates Bayesian inference to update event probabilities as new weather information arrives.

Rather than manually adjusting probabilities, the model updates beliefs using conditional probabilities associated with different weather signals.

This allows forecast probabilities to evolve in a statistically consistent manner as new information becomes available.

---

## Monte Carlo Backtesting

To evaluate robustness, the strategy is tested across many simulated market and weather-information paths.

Performance metrics include:

* Average Final PnL
* Win Rate
* PnL Standard Deviation
* Sharpe-Like Ratio
* Best Simulation Outcome
* Worst Simulation Outcome

Example Results:

* Average Final PnL ≈ 0.96
* Win Rate ≈ 84%
* Sharpe-Like Ratio ≈ 0.79

---

## Live Weather Forecast Integration

The simulator integrates live weather forecasts through the Open-Meteo API.

For each city, the framework retrieves:

* Hourly precipitation probabilities
* Maximum precipitation probability
* Average precipitation probability
* Time of highest precipitation risk

These forecasts are converted into prediction-market fair values and used to generate live trading recommendations.

---

## Multi-City Analysis

Supported cities:

* Philadelphia
* New York
* Chicago
* Miami

Example output:

| City         | Forecast Probability | Market Price | Signal   |
| ------------ | -------------------- | ------------ | -------- |
| Philadelphia | 3%                   | 10%          | SELL YES |
| New York     | 11%                  | 12%          | NO TRADE |
| Chicago      | 87%                  | 15%          | BUY YES  |
| Miami        | 32%                  | 20%          | BUY YES  |

This extension transforms the project from a single-contract simulator into a portfolio-style prediction market scanner.

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Requests
* Open-Meteo API

---

## Key Concepts Demonstrated

* Prediction Markets
* Probabilistic Forecasting
* Bayesian Updating
* Fair Value Estimation
* Trading Signal Generation
* Position Sizing
* Inventory Management
* Risk Controls
* Monte Carlo Simulation
* Performance Analytics
* Live Data Integration

---

## Future Improvements

Potential extensions include:

* Historical weather backtesting
* Real prediction-market data from Kalshi
* Real prediction-market data from Polymarket
* Multi-city portfolio optimization
* Weather market making strategies
* Forecast calibration analysis
* Additional weather contracts (snowfall, temperature, wind, hurricanes)
* Machine learning forecast models