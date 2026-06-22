from src.weather_contract import WeatherContract
from src.forecast_model import ForecastModel
from src.weather_api import get_tomorrow_rain_probability


contract = WeatherContract(
    event_name="Will Philadelphia receive rain tomorrow?",
    threshold=0.50,
    contract_type="rain_above",
    city="Philadelphia"
)

real_forecast_probability = get_tomorrow_rain_probability()

forecast_model = ForecastModel(
    initial_probability=real_forecast_probability
)

market_price = 0.10

edge = forecast_model.calculate_yes_edge(
    market_price
)

signal = forecast_model.generate_signal(
    market_price
)

print(contract.describe())
print()
print("Real Forecast Probability:", real_forecast_probability)
print("YES Fair Value:", forecast_model.get_yes_fair_value())
print("Market Price:", market_price)
print("Edge:", round(edge, 4))
print("Trading Signal:", signal)