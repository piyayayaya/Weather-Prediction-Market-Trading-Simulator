from weather_api import (
    get_philadelphia_forecast
)

forecast = get_philadelphia_forecast()

print(forecast["hourly"]["time"][:24])

print()

print(
    forecast["hourly"][
        "precipitation_probability"
    ][:24]
)