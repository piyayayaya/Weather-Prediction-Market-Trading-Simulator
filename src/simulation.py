import numpy as np


def simulate_market_prices(
    initial_price=0.60,
    num_steps=20,
    volatility=0.04,
    seed=42
):
    np.random.seed(seed)

    prices = [initial_price]

    for _ in range(num_steps - 1):
        price_change = np.random.normal(
            loc=0,
            scale=volatility
        )

        new_price = prices[-1] + price_change

        new_price = max(
            0.01,
            min(0.99, new_price)
        )

        prices.append(new_price)

    return prices


def simulate_weather_information_updates(
    num_steps=20,
    seed=7
):
    np.random.seed(seed)

    update_types = [
        "bullish_update",
        "bearish_update",
        "neutral_update"
    ]

    update_probabilities = [
        0.30,
        0.25,
        0.45
    ]

    updates = []

    for _ in range(num_steps):
        update_type = np.random.choice(
            update_types,
            p=update_probabilities
        )

        if update_type == "bullish_update":
            probability_change = np.random.uniform(
                0.02,
                0.08
            )

        elif update_type == "bearish_update":
            probability_change = np.random.uniform(
                -0.08,
                -0.02
            )

        else:
            probability_change = np.random.uniform(
                -0.01,
                0.01
            )

        updates.append(
            {
                "update_type": update_type,
                "probability_change": probability_change
            }
        )

    return updates


def build_forecast_probability_path(
    initial_probability,
    weather_updates
):
    probabilities = [initial_probability]

    for update in weather_updates[1:]:
        new_probability = probabilities[-1] + update["probability_change"]

        new_probability = max(
            0.01,
            min(0.99, new_probability)
        )

        probabilities.append(new_probability)

    return probabilities