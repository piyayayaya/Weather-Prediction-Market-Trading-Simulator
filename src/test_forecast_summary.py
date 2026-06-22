from weather_api import get_forecast_summary


summary = get_forecast_summary()

print("Forecast Summary")
print("Max Rain Probability:", summary["max_probability"])
print("Average Rain Probability:", summary["average_probability"])
print("Max Probability Time:", summary["max_probability_time"])