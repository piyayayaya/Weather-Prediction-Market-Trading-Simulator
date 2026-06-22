import requests


CITY_COORDINATES = {
    "Philadelphia": {
        "latitude": 39.9526,
        "longitude": -75.1652
    },
    "New York": {
        "latitude": 40.7128,
        "longitude": -74.0060
    },
    "Chicago": {
        "latitude": 41.8781,
        "longitude": -87.6298
    },
    "Miami": {
        "latitude": 25.7617,
        "longitude": -80.1918
    }
}


def get_forecast(city="Philadelphia"):
    if city not in CITY_COORDINATES:
        raise ValueError("City not supported")

    latitude = CITY_COORDINATES[city]["latitude"]
    longitude = CITY_COORDINATES[city]["longitude"]

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=precipitation_probability"
        "&forecast_days=2"
    )

    response = requests.get(url)
    data = response.json()

    return data


def get_forecast_summary(city="Philadelphia"):
    forecast = get_forecast(city)

    times = forecast["hourly"]["time"]
    probabilities = forecast["hourly"]["precipitation_probability"]

    max_probability = max(probabilities) / 100
    average_probability = sum(probabilities) / len(probabilities) / 100

    max_index = probabilities.index(max(probabilities))
    max_probability_time = times[max_index]

    return {
        "city": city,
        "max_probability": round(max_probability, 4),
        "average_probability": round(average_probability, 4),
        "max_probability_time": max_probability_time
    }


def get_tomorrow_rain_probability(city="Philadelphia"):
    summary = get_forecast_summary(city)

    return summary["max_probability"]