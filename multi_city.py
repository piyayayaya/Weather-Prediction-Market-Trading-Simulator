from src.forecast_model import ForecastModel
from src.trader import Trader
from src.weather_api import get_forecast_summary


cities = [
    "Philadelphia",
    "New York",
    "Chicago",
    "Miami"
]

market_prices = {
    "Philadelphia": 0.10,
    "New York": 0.12,
    "Chicago": 0.15,
    "Miami": 0.20
}

trader = Trader(
    initial_cash=1000.0,
    max_long_inventory=5,
    max_short_inventory=-5
)

print("Multi-City Weather Prediction Market Signals")
print()

for city in cities:
    forecast_summary = get_forecast_summary(city)

    real_forecast_probability = forecast_summary["max_probability"]

    forecast_model = ForecastModel(
        initial_probability=real_forecast_probability
    )

    market_price = market_prices[city]

    edge = forecast_model.calculate_yes_edge(
        market_price
    )

    signal = forecast_model.generate_signal(
        market_price
    )

    quantity = forecast_model.calculate_position_size(
        market_price
    )

    if signal == "BUY YES" and quantity > 0:
        trade_allowed = trader.can_buy_yes(quantity)

    elif signal == "SELL YES" and quantity > 0:
        trade_allowed = trader.can_sell_yes(quantity)

    else:
        trade_allowed = False

    print("City:", city)
    print("Max Rain Probability:", forecast_summary["max_probability"])
    print("Average Rain Probability:", forecast_summary["average_probability"])
    print("Max Probability Time:", forecast_summary["max_probability_time"])
    print("Market Price:", market_price)
    print("Model Fair Value:", forecast_model.get_yes_fair_value())
    print("Edge:", round(edge, 4))
    print("Signal:", signal)
    print("Quantity:", quantity)
    print("Trade Allowed:", trade_allowed)
    print()