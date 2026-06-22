import numpy as np

from src.weather_contract import WeatherContract
from src.forecast_model import ForecastModel
from src.trader import Trader
from src.simulation import (
    simulate_market_prices,
    simulate_weather_information_updates
)
from src.bayes import bayesian_update
from src.weather_signal import get_signal_likelihoods
from src.analytics import generate_summary_statistics


def run_single_simulation(
    market_seed,
    weather_seed,
    num_steps=10,
    initial_capital=1000.0
):
    contract = WeatherContract(
        event_name="Will Philadelphia receive more than 0.50 inches of rain tomorrow?",
        threshold=0.50,
        contract_type="rain_above",
        city="Philadelphia"
    )

    forecast_model = ForecastModel(
        initial_probability=0.72
    )

    trader = Trader(
        initial_cash=initial_capital,
        max_long_inventory=5,
        max_short_inventory=0
    )

    market_prices = simulate_market_prices(
        initial_price=0.60,
        num_steps=num_steps,
        volatility=0.04,
        seed=market_seed
    )

    weather_updates = simulate_weather_information_updates(
        num_steps=num_steps,
        seed=weather_seed
    )

    trade_log = []

    for time_step, market_price in enumerate(market_prices):
        update_type = weather_updates[time_step]["update_type"]

        likelihood_if_event, likelihood_if_no_event = get_signal_likelihoods(
            update_type
        )

        posterior_probability = bayesian_update(
            prior=forecast_model.probability,
            likelihood_if_event=likelihood_if_event,
            likelihood_if_no_event=likelihood_if_no_event
        )

        forecast_model.update_probability(
            posterior_probability
        )

        edge = forecast_model.calculate_yes_edge(
            market_price
        )

        signal = forecast_model.generate_signal(
            market_price
        )

        quantity = forecast_model.calculate_position_size(
            market_price
        )

        trade_executed = False
        trade_blocked_by_risk_limit = False

        if signal == "BUY YES" and quantity > 0:
            trade_executed = trader.buy_yes(
                market_price,
                quantity=quantity
            )

            if not trade_executed:
                trade_blocked_by_risk_limit = True

        elif signal == "SELL YES" and quantity > 0:
            trade_executed = trader.sell_yes(
                market_price,
                quantity=quantity
            )

            if not trade_executed:
                trade_blocked_by_risk_limit = True

        portfolio_value = trader.get_total_value(
            market_price
        )

        pnl = portfolio_value - initial_capital

        row = {
            "time_step": time_step,
            "market_price": round(market_price, 4),
            "weather_update": update_type,
            "probability_change": round(
                weather_updates[time_step]["probability_change"],
                4
            ),
            "model_probability": round(forecast_model.probability, 4),
            "edge": round(edge, 4),
            "signal": signal,
            "quantity": quantity,
            "trade_executed": trade_executed,
            "trade_blocked_by_risk_limit": trade_blocked_by_risk_limit,
            "cash": round(trader.cash, 2),
            "yes_inventory": trader.yes_inventory,
            "portfolio_value": round(portfolio_value, 2),
            "pnl": round(pnl, 2)
        }

        trade_log.append(row)

    np.random.seed(
        market_seed + weather_seed
    )

    final_event_probability = forecast_model.probability
    event_occurs = np.random.random() < final_event_probability

    if event_occurs:
        realized_rainfall = 0.72
    else:
        realized_rainfall = 0.20

    contract.resolve(
        realized_rainfall
    )

    final_payout_per_contract = contract.payout(
        "YES"
    )

    final_contract_value = trader.yes_inventory * final_payout_per_contract
    final_portfolio_value = trader.cash + final_contract_value
    final_pnl = final_portfolio_value - initial_capital

    stats = generate_summary_statistics(
        trade_log
    )

    return {
        "final_pnl": round(final_pnl, 2),
        "event_occurs": event_occurs,
        "final_event_probability": round(final_event_probability, 4),
        "realized_rainfall": realized_rainfall,
        "total_trades": stats["Total Trades"],
        "blocked_trades": stats["Blocked Trades"],
        "max_inventory": stats["Max Inventory"],
        "average_edge": stats["Average Edge"]
    }


def run_monte_carlo_backtest(
    num_simulations=100
):
    results = []

    for simulation_id in range(num_simulations):
        result = run_single_simulation(
            market_seed=simulation_id,
            weather_seed=simulation_id + 100
        )

        result["simulation_id"] = simulation_id
        results.append(result)

    return results