# Weather Prediction-Market Trading Simulator

This project is a Python simulator for trading binary weather prediction markets.

The main contract I use is:

> Will Philadelphia receive at least 0.50 inches of rain tomorrow?

A YES contract pays $1 if the event happens and $0 otherwise.  
A NO contract pays the opposite.

The goal of the project is to take weather forecast data, convert it into event probabilities, use those probabilities as fair values for YES/NO contracts, and then simulate trading around differences between model value and market price.

---

## Project Overview

The project has a few main parts:

- pulls historical and live weather data from Open-Meteo
- measures historical forecast errors
- converts forecasts into event probabilities
- calibrates those probabilities using walk-forward validation
- updates probabilities as forecasts change during the day
- compares model fair value to market bid/ask prices
- sizes trades using fractional Kelly sizing
- tracks positions, cash, and P&L
- settles contracts at $0 or $1
- runs Monte Carlo simulations
- runs control tests and sensitivity analysis

---

## Data

The project uses Open-Meteo data for Philadelphia.

I use:

- historical precipitation forecasts
- historical realized precipitation
- live precipitation forecasts

The historical sample currently contains 90 aligned forecast/actual observations.

The event threshold is:

0.50 inches of rain