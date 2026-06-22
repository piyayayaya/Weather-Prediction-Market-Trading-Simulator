from weather_api import (
    get_tomorrow_rain_probability
)

probability = (
    get_tomorrow_rain_probability()
)

print(
    "Real Forecast Probability:",
    probability
)