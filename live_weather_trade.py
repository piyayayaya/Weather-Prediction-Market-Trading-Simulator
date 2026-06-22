from src.weather_contract import WeatherContract
from src.forecast_model import ForecastModel
from src.trader import Trader
from src.weather_api import get_forecast_summary


city = "Philadelphia"
market_price = 0.10

contract = WeatherContract(
    event_name=f"Will {city} receive rain tomorrow?",
    threshold=0.50,
    contract_type="rain_above",
    city=city
)

forecast_summary = get_forecast_summary(city)

real_forecast_probability = forecast_summary["max_probability"]

forecast_model = ForecastModel(
    initial_probability=real_forecast_probability
)

conservative_trader = Trader(
    initial_cash=1000.0,
    max_long_inventory=5,
    max_short_inventory=0
)

aggressive_trader = Trader(
    initial_cash=1000.0,
    max_long_inventory=5,
    max_short_inventory=-5
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


def check_trade_allowed(trader):
    if signal == "BUY YES" and quantity > 0:
        return trader.can_buy_yes(quantity)

    if signal == "SELL YES" and quantity > 0:
        return trader.can_sell_yes(quantity)

    return False


conservative_allowed = check_trade_allowed(
    conservative_trader
)

aggressive_allowed = check_trade_allowed(
    aggressive_trader
)

print(contract.describe())
print()

print("Forecast Summary")
print("City:", forecast_summary["city"])
print("Max Rain Probability:", forecast_summary["max_probability"])
print("Average Rain Probability:", forecast_summary["average_probability"])
print("Max Probability Time:", forecast_summary["max_probability_time"])
print()

print("Market Price:", market_price)
print("Model Fair Value:", forecast_model.get_yes_fair_value())
print("Edge:", round(edge, 4))
print("Trading Signal:", signal)
print("Suggested Quantity:", quantity)
print()

print("Risk Mode Comparison")
print("Conservative Mode Allows Trade:", conservative_allowed)
print("Aggressive Mode Allows Trade:", aggressive_allowed)