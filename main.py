from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import random
import statistics
from typing import Dict, List, Tuple

import requests
from scipy.stats import norm

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ==================================================
# FINAL PROJECT CONFIGURATION
# ==================================================

LAT = 39.9526
LON = -75.1652
TZ = "America/New_York"

# Fixed research sample for reproducibility.
HISTORICAL_START_DATE = "2026-06-19"
HISTORICAL_END_DATE = "2026-09-16"

RAIN_THRESHOLD = 0.50
STARTING_CASH = 100.0

# Probability model.
OUTER_VALIDATION_TRAINING_DAYS = 45
INNER_CALIBRATION_TRAINING_DAYS = 20
REVISION_WEIGHT = 0.50

# Final trading specification.
EDGE_THRESHOLD = 0.08
POSITION_CAP = 50
KELLY_FRACTION = 0.50
MAX_CASH_FRACTION = 0.40
EXIT_BUFFER = 0.02

# Baseline synthetic market assumptions.
# These assumptions are the explicit source of simulated informational edge.
BASELINE_MARKET_REACTION_SPEED = 0.30
BASELINE_MARKET_ERROR_MULTIPLIER = 1.35
BASELINE_MARKET_NOISE_SD = 0.015
BASELINE_HALF_SPREAD = 0.010

# Monte Carlo.
MONTE_CARLO_SIMULATIONS = 1000
RANDOM_SEED = 42

# Stress-test grid.
REACTION_SPEED_GRID = [0.20, 0.30, 0.40, 0.50, 0.70, 1.00]
MARKET_NOISE_GRID = [0.04, 0.03, 0.02, 0.01, 0.00]
KELLY_FRACTION_GRID = [0.25, 0.50, 0.75]

RESULTS_DIR = Path("results")


# ==================================================
# GENERAL HELPERS
# ==================================================


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def ensure_results_directory() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# WEATHER DATA
# ==================================================


def get_tomorrow_forecast() -> Tuple[str, float]:
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": "precipitation_sum",
        "precipitation_unit": "inch",
        "timezone": TZ,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    daily = response.json()["daily"]

    tomorrow = (
        datetime.now(ZoneInfo(TZ)).date() + timedelta(days=1)
    ).isoformat()

    if tomorrow not in daily["time"]:
        raise ValueError(f"{tomorrow} not found in live forecast.")

    index = daily["time"].index(tomorrow)
    forecast = daily["precipitation_sum"][index]

    if forecast is None:
        raise ValueError("Tomorrow's precipitation forecast is missing.")

    return tomorrow, float(forecast)


def get_historical_actuals(
    start_date: str,
    end_date: str,
) -> Tuple[List[str], List[float]]:
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "precipitation_unit": "inch",
        "timezone": TZ,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    daily = response.json()["daily"]

    return daily["time"], daily["precipitation_sum"]


def get_historical_forecasts(
    start_date: str,
    end_date: str,
) -> Tuple[List[str], List[float]]:
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "precipitation_previous_day1",
        "start_date": start_date,
        "end_date": end_date,
        "precipitation_unit": "inch",
        "timezone": TZ,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for timestamp, value in zip(
        hourly["time"],
        hourly["precipitation_previous_day1"],
    ):
        if value is None:
            continue

        date = timestamp[:10]
        totals[date] = totals.get(date, 0.0) + float(value)
        counts[date] = counts.get(date, 0) + 1

    dates: List[str] = []
    forecasts: List[float] = []

    for date in sorted(totals):
        if counts[date] != 24:
            continue

        dates.append(date)
        forecasts.append(totals[date])

    return dates, forecasts


def align_historical_data(
    forecast_dates: List[str],
    forecasts: List[float],
    actual_dates: List[str],
    actuals: List[float],
) -> Tuple[List[str], List[float], List[float]]:
    actual_by_date = dict(zip(actual_dates, actuals))
    rows = []

    for date, forecast in zip(forecast_dates, forecasts):
        if date not in actual_by_date:
            continue

        actual = actual_by_date[date]

        if forecast is None or actual is None:
            continue

        rows.append((date, float(forecast), float(actual)))

    rows.sort(key=lambda row: row[0])

    return (
        [row[0] for row in rows],
        [row[1] for row in rows],
        [row[2] for row in rows],
    )


# ==================================================
# HISTORICAL STATISTICS
# ==================================================


def calculate_event_base_rate(
    actuals: List[float],
    threshold: float,
) -> Tuple[float, int, int]:
    if not actuals:
        raise ValueError("Historical actuals cannot be empty.")

    yes_days = sum(actual >= threshold for actual in actuals)
    no_days = len(actuals) - yes_days
    base_rate = yes_days / len(actuals)

    return base_rate, yes_days, no_days


def analyze_forecast_errors(
    forecasts: List[float],
    actuals: List[float],
) -> Tuple[float, float, List[float]]:
    if len(forecasts) != len(actuals):
        raise ValueError("Forecasts and actuals must have equal length.")

    errors = [actual - forecast for forecast, actual in zip(forecasts, actuals)]

    if len(errors) < 2:
        raise ValueError("At least two historical observations are required.")

    return statistics.mean(errors), statistics.stdev(errors), errors


# ==================================================
# WEATHER PROBABILITY MODELS
# ==================================================


def empirical_rainfall_probability(
    forecast: float,
    historical_errors: List[float],
    threshold: float,
) -> float:
    if not historical_errors:
        raise ValueError("Historical errors cannot be empty.")

    successes = 0

    for error in historical_errors:
        simulated_actual = max(0.0, forecast + error)

        if simulated_actual >= threshold:
            successes += 1

    return successes / len(historical_errors)


def normal_rainfall_probability(
    corrected_forecast: float,
    error_sd: float,
    threshold: float,
) -> float:
    if error_sd <= 0:
        raise ValueError("Forecast error SD must be positive.")

    z_score = (threshold - corrected_forecast) / error_sd
    return float(1.0 - norm.cdf(z_score))


# ==================================================
# PROBABILITY CALIBRATION
# ==================================================


def shrink_probability(
    raw_probability: float,
    base_rate: float,
    shrinkage_lambda: float,
) -> float:
    calibrated = (
        (1.0 - shrinkage_lambda) * base_rate
        + shrinkage_lambda * raw_probability
    )

    return clamp(calibrated, 1e-6, 1.0 - 1e-6)


def calculate_brier_score(
    probabilities: List[float],
    outcomes: List[int],
) -> float:
    if len(probabilities) != len(outcomes):
        raise ValueError("Probabilities and outcomes must have equal length.")

    if not probabilities:
        raise ValueError("No predictions available for Brier score.")

    return sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes)
    ) / len(probabilities)


def choose_shrinkage_lambda(
    forecasts: List[float],
    actuals: List[float],
    threshold: float,
    minimum_training_days: int = INNER_CALIBRATION_TRAINING_DAYS,
) -> Tuple[float, Dict[float, float]]:
    lambda_grid = [i / 10 for i in range(11)]

    if len(forecasts) <= minimum_training_days:
        return 0.0, {}

    candidate_predictions = {
        candidate_lambda: []
        for candidate_lambda in lambda_grid
    }
    outcomes: List[int] = []

    for test_index in range(minimum_training_days, len(forecasts)):
        training_forecasts = forecasts[:test_index]
        training_actuals = actuals[:test_index]

        training_base_rate, _, _ = calculate_event_base_rate(
            training_actuals,
            threshold,
        )

        _, _, training_errors = analyze_forecast_errors(
            training_forecasts,
            training_actuals,
        )

        raw_probability = empirical_rainfall_probability(
            forecasts[test_index],
            training_errors,
            threshold,
        )

        outcome = int(actuals[test_index] >= threshold)
        outcomes.append(outcome)

        for candidate_lambda in lambda_grid:
            candidate_predictions[candidate_lambda].append(
                shrink_probability(
                    raw_probability,
                    training_base_rate,
                    candidate_lambda,
                )
            )

    lambda_scores = {
        candidate_lambda: calculate_brier_score(
            candidate_predictions[candidate_lambda],
            outcomes,
        )
        for candidate_lambda in lambda_grid
    }

    best_lambda = min(
        lambda_scores,
        key=lambda candidate_lambda: (
            lambda_scores[candidate_lambda],
            candidate_lambda,
        ),
    )

    return best_lambda, lambda_scores


def build_calibration_table(
    probabilities: List[float],
    outcomes: List[int],
    number_of_bins: int = 10,
) -> Tuple[List[Dict[str, float]], float]:
    bins = [
        {
            "lower": i / number_of_bins,
            "upper": (i + 1) / number_of_bins,
            "count": 0,
            "prediction_sum": 0.0,
            "outcome_sum": 0.0,
        }
        for i in range(number_of_bins)
    ]

    for probability, outcome in zip(probabilities, outcomes):
        bin_index = min(int(probability * number_of_bins), number_of_bins - 1)
        bins[bin_index]["count"] += 1
        bins[bin_index]["prediction_sum"] += probability
        bins[bin_index]["outcome_sum"] += outcome

    rows = []
    expected_calibration_error = 0.0

    for current_bin in bins:
        count = int(current_bin["count"])

        if count == 0:
            continue

        average_prediction = current_bin["prediction_sum"] / count
        observed_rate = current_bin["outcome_sum"] / count
        gap = average_prediction - observed_rate

        expected_calibration_error += (
            count / len(probabilities)
        ) * abs(gap)

        rows.append(
            {
                "lower": current_bin["lower"],
                "upper": current_bin["upper"],
                "count": count,
                "average_prediction": average_prediction,
                "observed_rate": observed_rate,
                "gap": gap,
            }
        )

    return rows, expected_calibration_error


def run_calibrated_walk_forward_validation(
    dates: List[str],
    forecasts: List[float],
    actuals: List[float],
    threshold: float,
    minimum_training_days: int = OUTER_VALIDATION_TRAINING_DAYS,
) -> Dict:
    if not (len(dates) == len(forecasts) == len(actuals)):
        raise ValueError("Dates, forecasts, and actuals must have equal length.")

    baseline_predictions: List[float] = []
    raw_predictions: List[float] = []
    calibrated_predictions: List[float] = []
    outcomes: List[int] = []
    selected_lambdas: List[float] = []
    rows = []

    for test_index in range(minimum_training_days, len(forecasts)):
        training_forecasts = forecasts[:test_index]
        training_actuals = actuals[:test_index]

        training_base_rate, _, _ = calculate_event_base_rate(
            training_actuals,
            threshold,
        )

        _, _, training_errors = analyze_forecast_errors(
            training_forecasts,
            training_actuals,
        )

        selected_lambda, _ = choose_shrinkage_lambda(
            training_forecasts,
            training_actuals,
            threshold,
            INNER_CALIBRATION_TRAINING_DAYS,
        )

        raw_probability = empirical_rainfall_probability(
            forecasts[test_index],
            training_errors,
            threshold,
        )

        calibrated_probability = shrink_probability(
            raw_probability,
            training_base_rate,
            selected_lambda,
        )

        outcome = int(actuals[test_index] >= threshold)

        baseline_predictions.append(training_base_rate)
        raw_predictions.append(raw_probability)
        calibrated_predictions.append(calibrated_probability)
        outcomes.append(outcome)
        selected_lambdas.append(selected_lambda)

        rows.append(
            {
                "date": dates[test_index],
                "forecast": forecasts[test_index],
                "actual": actuals[test_index],
                "outcome": outcome,
                "base_rate_probability": training_base_rate,
                "raw_empirical_probability": raw_probability,
                "selected_lambda": selected_lambda,
                "calibrated_probability": calibrated_probability,
            }
        )

    baseline_brier = calculate_brier_score(baseline_predictions, outcomes)
    raw_brier = calculate_brier_score(raw_predictions, outcomes)
    calibrated_brier = calculate_brier_score(calibrated_predictions, outcomes)

    calibrated_skill = 1.0 - calibrated_brier / baseline_brier

    calibration_rows, calibrated_ece = build_calibration_table(
        calibrated_predictions,
        outcomes,
    )

    return {
        "predictions": len(outcomes),
        "yes_days": sum(outcomes),
        "no_days": len(outcomes) - sum(outcomes),
        "baseline_brier": baseline_brier,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "calibrated_skill": calibrated_skill,
        "average_lambda": statistics.mean(selected_lambdas),
        "selected_lambdas": selected_lambdas,
        "calibrated_ece": calibrated_ece,
        "calibration_rows": calibration_rows,
        "rows": rows,
        "baseline_predictions": baseline_predictions,
        "raw_predictions": raw_predictions,
        "calibrated_predictions": calibrated_predictions,
        "outcomes": outcomes,
    }


# ==================================================
# BAYESIAN REVISION MODEL
# ==================================================


def probability_to_odds(probability: float) -> float:
    probability = clamp(probability, 1e-6, 1.0 - 1e-6)
    return probability / (1.0 - probability)


def odds_to_probability(odds: float) -> float:
    return odds / (1.0 + odds)


def bayesian_revision_update(
    prior_yes: float,
    current_signal: float,
    previous_signal: float | None,
    base_rate: float,
    revision_weight: float,
) -> float:
    prior_odds = probability_to_odds(prior_yes)
    current_signal_odds = probability_to_odds(current_signal)

    if previous_signal is None:
        reference_odds = probability_to_odds(base_rate)
        applied_weight = 1.0
    else:
        reference_odds = probability_to_odds(previous_signal)
        applied_weight = revision_weight

    raw_revision_ratio = current_signal_odds / reference_odds
    tempered_revision_ratio = raw_revision_ratio ** applied_weight
    posterior_odds = prior_odds * tempered_revision_ratio

    return odds_to_probability(posterior_odds)


# ==================================================
# MARKET HELPERS
# ==================================================


def make_yes_market(
    midpoint: float,
    half_spread: float,
) -> Tuple[float, float]:
    midpoint = clamp(midpoint, 0.02, 0.98)

    yes_bid = max(0.01, midpoint - half_spread)
    yes_ask = min(0.99, midpoint + half_spread)

    if yes_ask <= yes_bid:
        yes_ask = min(0.99, yes_bid + 0.01)

    return yes_bid, yes_ask


# ==================================================
# POSITION ACCOUNTING
# ==================================================


def execute_buy(
    side: str,
    size: int,
    price: float,
    yes_position: int,
    no_position: int,
    yes_average_cost: float,
    no_average_cost: float,
    cash: float,
) -> Tuple[int, int, float, float, float]:
    if size <= 0:
        return (
            yes_position,
            no_position,
            yes_average_cost,
            no_average_cost,
            cash,
        )

    cost = size * price

    if side == "YES":
        existing_cost = yes_position * yes_average_cost
        yes_position += size
        yes_average_cost = (existing_cost + cost) / yes_position

    elif side == "NO":
        existing_cost = no_position * no_average_cost
        no_position += size
        no_average_cost = (existing_cost + cost) / no_position

    else:
        raise ValueError("Side must be YES or NO.")

    cash -= cost

    return (
        yes_position,
        no_position,
        yes_average_cost,
        no_average_cost,
        cash,
    )


def execute_sell(
    side: str,
    size: int,
    price: float,
    yes_position: int,
    no_position: int,
    yes_average_cost: float,
    no_average_cost: float,
    cash: float,
) -> Tuple[int, int, float, float, float]:
    if size <= 0:
        return (
            yes_position,
            no_position,
            yes_average_cost,
            no_average_cost,
            cash,
        )

    if side == "YES":
        size = min(size, yes_position)
        cash += size * price
        yes_position -= size

        if yes_position == 0:
            yes_average_cost = 0.0

    elif side == "NO":
        size = min(size, no_position)
        cash += size * price
        no_position -= size

        if no_position == 0:
            no_average_cost = 0.0

    else:
        raise ValueError("Side must be YES or NO.")

    return (
        yes_position,
        no_position,
        yes_average_cost,
        no_average_cost,
        cash,
    )


# ==================================================
# KELLY POSITION SIZING
# ==================================================


def calculate_kelly_size(
    probability: float,
    price: float,
    cash: float,
    current_position: int,
    max_position: int,
    kelly_fraction: float,
    max_cash_fraction: float = MAX_CASH_FRACTION,
) -> int:
    edge = probability - price

    if edge <= 0 or price <= 0 or price >= 1:
        return 0

    full_kelly_fraction = edge / (1.0 - price)
    fraction_to_use = min(
        kelly_fraction * full_kelly_fraction,
        max_cash_fraction,
    )

    capital_to_use = cash * fraction_to_use
    affordable_contracts = int(capital_to_use / price)
    remaining_capacity = max_position - current_position

    return max(
        0,
        min(affordable_contracts, remaining_capacity),
    )


# ==================================================
# SETTLEMENT-BASED TRADING STRATEGY
# ==================================================


def run_settlement_trading_day(
    updates: List[Dict],
    actual_outcome: int,
    base_rate: float,
    historical_errors: List[float],
    threshold: float,
    shrinkage_lambda: float,
    revision_weight: float = REVISION_WEIGHT,
    edge_threshold: float = EDGE_THRESHOLD,
    max_position: int = POSITION_CAP,
    starting_cash: float = STARTING_CASH,
    exit_buffer: float = EXIT_BUFFER,
    kelly_fraction: float = KELLY_FRACTION,
    market_mode: str = "provided",
    control_half_spread: float = BASELINE_HALF_SPREAD,
    efficiency_reaction_speed: float = 1.0,
    efficiency_noise_sd: float = 0.0,
) -> Dict:
    cash = starting_cash

    yes_position = 0
    no_position = 0
    yes_average_cost = 0.0
    no_average_cost = 0.0

    current_prior_yes = base_rate
    previous_calibrated_signal = None

    account_values = [starting_cash]
    transaction_count = 0
    hit_position_cap = False
    trade_log = []
    previous_efficiency_midpoint = base_rate

    for update_number, update in enumerate(updates, start=1):
        forecast = update["rainfall_forecast"]

        raw_probability = empirical_rainfall_probability(
            forecast,
            historical_errors,
            threshold,
        )

        calibrated_signal = shrink_probability(
            raw_probability,
            base_rate,
            shrinkage_lambda,
        )

        posterior_yes = bayesian_revision_update(
            current_prior_yes,
            calibrated_signal,
            previous_calibrated_signal,
            base_rate,
            revision_weight,
        )

        current_prior_yes = posterior_yes
        previous_calibrated_signal = calibrated_signal

        yes_fair = posterior_yes
        no_fair = 1.0 - posterior_yes

        if market_mode == "efficient_control":
            yes_bid, yes_ask = make_yes_market(
                yes_fair,
                control_half_spread,
            )
        elif market_mode == "fair_value_convergence":
            noise_z = update.get("market_noise_z", 0.0)
            market_midpoint = (
                previous_efficiency_midpoint
                + efficiency_reaction_speed
                * (yes_fair - previous_efficiency_midpoint)
                + efficiency_noise_sd * noise_z
            )
            market_midpoint = clamp(market_midpoint, 0.02, 0.98)
            yes_bid, yes_ask = make_yes_market(
                market_midpoint,
                control_half_spread,
            )
            previous_efficiency_midpoint = market_midpoint
        elif market_mode == "provided":
            yes_bid = update["yes_bid"]
            yes_ask = update["yes_ask"]
        else:
            raise ValueError("Unknown market_mode.")

        no_bid = 1.0 - yes_ask
        no_ask = 1.0 - yes_bid

        actions = []

        # Model-based exits before adding new risk.
        if yes_position > 0 and yes_bid > yes_fair + exit_buffer:
            size = yes_position
            (
                yes_position,
                no_position,
                yes_average_cost,
                no_average_cost,
                cash,
            ) = execute_sell(
                "YES",
                size,
                yes_bid,
                yes_position,
                no_position,
                yes_average_cost,
                no_average_cost,
                cash,
            )
            transaction_count += 1
            actions.append(f"SELL YES {size} @ {yes_bid:.3f}")

        if no_position > 0 and no_bid > no_fair + exit_buffer:
            size = no_position
            (
                yes_position,
                no_position,
                yes_average_cost,
                no_average_cost,
                cash,
            ) = execute_sell(
                "NO",
                size,
                no_bid,
                yes_position,
                no_position,
                yes_average_cost,
                no_average_cost,
                cash,
            )
            transaction_count += 1
            actions.append(f"SELL NO {size} @ {no_bid:.3f}")

        yes_edge = yes_fair - yes_ask
        no_edge = no_fair - no_ask

        # Open/add only when executable edge clears threshold.
        if max(yes_edge, no_edge) > edge_threshold:
            if yes_edge >= no_edge:
                chosen_side = "YES"
                chosen_probability = yes_fair
                chosen_price = yes_ask

                # Net out opposing exposure first.
                if no_position > 0:
                    size = no_position
                    (
                        yes_position,
                        no_position,
                        yes_average_cost,
                        no_average_cost,
                        cash,
                    ) = execute_sell(
                        "NO",
                        size,
                        no_bid,
                        yes_position,
                        no_position,
                        yes_average_cost,
                        no_average_cost,
                        cash,
                    )
                    transaction_count += 1
                    actions.append(f"NET OUT NO {size} @ {no_bid:.3f}")

                current_position = yes_position

            else:
                chosen_side = "NO"
                chosen_probability = no_fair
                chosen_price = no_ask

                if yes_position > 0:
                    size = yes_position
                    (
                        yes_position,
                        no_position,
                        yes_average_cost,
                        no_average_cost,
                        cash,
                    ) = execute_sell(
                        "YES",
                        size,
                        yes_bid,
                        yes_position,
                        no_position,
                        yes_average_cost,
                        no_average_cost,
                        cash,
                    )
                    transaction_count += 1
                    actions.append(f"NET OUT YES {size} @ {yes_bid:.3f}")

                current_position = no_position

            size = calculate_kelly_size(
                chosen_probability,
                chosen_price,
                cash,
                current_position,
                max_position,
                kelly_fraction,
            )

            if size > 0:
                (
                    yes_position,
                    no_position,
                    yes_average_cost,
                    no_average_cost,
                    cash,
                ) = execute_buy(
                    chosen_side,
                    size,
                    chosen_price,
                    yes_position,
                    no_position,
                    yes_average_cost,
                    no_average_cost,
                    cash,
                )
                transaction_count += 1
                actions.append(
                    f"BUY {chosen_side} {size} @ {chosen_price:.3f}"
                )

        if max(yes_position, no_position) >= max_position:
            hit_position_cap = True

        account_value = (
            cash
            + yes_position * yes_bid
            + no_position * no_bid
        )

        account_values.append(account_value)

        trade_log.append(
            {
                "update": update_number,
                "time": update["time"],
                "forecast": forecast,
                "raw_probability": raw_probability,
                "calibrated_signal": calibrated_signal,
                "posterior_yes": posterior_yes,
                "yes_fair": yes_fair,
                "no_fair": no_fair,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_bid": no_bid,
                "no_ask": no_ask,
                "yes_edge": yes_edge,
                "no_edge": no_edge,
                "actions": actions,
                "yes_position": yes_position,
                "no_position": no_position,
                "cash": cash,
                "account_value": account_value,
            }
        )

    # Event settlement: winning contract pays $1, losing contract pays $0.
    yes_payout = actual_outcome
    no_payout = 1 - actual_outcome

    settlement_value = (
        yes_position * yes_payout
        + no_position * no_payout
    )

    cash += settlement_value
    final_pnl = cash - starting_cash

    account_values.append(cash)

    peak = account_values[0]
    maximum_drawdown = 0.0

    for account_value in account_values:
        peak = max(peak, account_value)
        maximum_drawdown = max(
            maximum_drawdown,
            peak - account_value,
        )

    return {
        "final_pnl": final_pnl,
        "final_cash": cash,
        "transaction_count": transaction_count,
        "hit_position_cap": hit_position_cap,
        "max_drawdown": maximum_drawdown,
        "trade_log": trade_log,
        "settlement_yes": actual_outcome,
    }


# ==================================================
# SYNTHETIC MARKET / MONTE CARLO
# ==================================================


def evolve_error(
    previous_error: float,
    mean_error: float,
    previous_sd: float,
    target_sd: float,
    persistence: float,
    rng: random.Random,
) -> float:
    scale_ratio = target_sd / previous_sd

    persistent_component = (
        persistence
        * (previous_error - mean_error)
        * scale_ratio
    )

    innovation_sd = target_sd * (1.0 - persistence ** 2) ** 0.5
    innovation = rng.gauss(0.0, innovation_sd)

    return mean_error + persistent_component + innovation


def generate_scenario(
    historical_actuals: List[float],
    historical_errors: List[float],
    mean_error: float,
    error_sd: float,
    threshold: float,
    base_rate: float,
    rng: random.Random,
    market_reaction_speed: float,
    market_error_multiplier: float,
    market_noise_sd: float,
    half_spread: float,
    strategy_persistence: float = 0.70,
    market_persistence: float = 0.75,
) -> Dict:
    times = ["8:00 AM", "12:00 PM", "4:00 PM", "8:00 PM"]
    error_scales = [1.00, 0.75, 0.55, 0.40]

    # Eventual true rainfall is sampled from the historical distribution.
    index = rng.randrange(len(historical_actuals))
    latent_actual = historical_actuals[index]
    actual_outcome = int(latent_actual >= threshold)

    # Trader gets the historical forecast-error draw associated with that day.
    strategy_error = historical_errors[index]
    strategy_previous_sd = error_sd

    # Market gets an independent, potentially noisier signal.
    sampled_market_error = rng.choice(historical_errors)
    market_error = (
        mean_error
        + market_error_multiplier
        * (sampled_market_error - mean_error)
    )
    market_previous_sd = error_sd * market_error_multiplier

    # Market begins anchored near the base rate.
    previous_market_midpoint = base_rate
    updates = []

    for update_index, time_label in enumerate(times):
        strategy_target_sd = error_sd * error_scales[update_index]
        market_target_sd = (
            error_sd
            * market_error_multiplier
            * error_scales[update_index]
        )

        if update_index > 0:
            strategy_error = evolve_error(
                strategy_error,
                mean_error,
                strategy_previous_sd,
                strategy_target_sd,
                strategy_persistence,
                rng,
            )

            market_error = evolve_error(
                market_error,
                mean_error,
                market_previous_sd,
                market_target_sd,
                market_persistence,
                rng,
            )

        strategy_forecast = max(0.0, latent_actual - strategy_error)

        market_weather_estimate = max(0.0, latent_actual - market_error)
        corrected_market_estimate = max(
            0.0,
            market_weather_estimate + mean_error,
        )

        market_target_probability = normal_rainfall_probability(
            corrected_market_estimate,
            error_sd * market_error_multiplier,
            threshold,
        )

        # Explicit simulated inefficiency:
        # market updates only partially toward its latest signal.
        market_noise_shock = rng.gauss(0.0, market_noise_sd)
        market_noise_z = (
            market_noise_shock / market_noise_sd
            if market_noise_sd > 0
            else 0.0
        )

        market_midpoint = (
            previous_market_midpoint
            + market_reaction_speed
            * (market_target_probability - previous_market_midpoint)
            + market_noise_shock
        )

        market_midpoint = clamp(market_midpoint, 0.02, 0.98)
        yes_bid, yes_ask = make_yes_market(market_midpoint, half_spread)

        updates.append(
            {
                "time": time_label,
                "rainfall_forecast": strategy_forecast,
                "market_midpoint": market_midpoint,
                "market_noise_z": market_noise_z,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
            }
        )

        previous_market_midpoint = market_midpoint
        strategy_previous_sd = strategy_target_sd
        market_previous_sd = market_target_sd

    return {
        "actual_outcome": actual_outcome,
        "actual_rainfall": latent_actual,
        "updates": updates,
    }


def percentile(values: List[float], probability: float) -> float:
    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower

    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_monte_carlo(
    pnl_values: List[float],
    drawdowns: List[float],
    transaction_counts: List[int],
    cap_hits: int,
) -> Dict:
    return {
        "simulations": len(pnl_values),
        "mean_pnl": statistics.mean(pnl_values),
        "median_pnl": statistics.median(pnl_values),
        "pnl_sd": statistics.stdev(pnl_values) if len(pnl_values) > 1 else 0.0,
        "win_rate": sum(pnl > 0 for pnl in pnl_values) / len(pnl_values),
        "loss_rate": sum(pnl < 0 for pnl in pnl_values) / len(pnl_values),
        "flat_rate": sum(abs(pnl) < 1e-12 for pnl in pnl_values) / len(pnl_values),
        "p05": percentile(pnl_values, 0.05),
        "p95": percentile(pnl_values, 0.95),
        "worst": min(pnl_values),
        "best": max(pnl_values),
        "average_drawdown": statistics.mean(drawdowns),
        "p95_drawdown": percentile(drawdowns, 0.95),
        "average_transactions": statistics.mean(transaction_counts),
        "cap_hit_rate": cap_hits / len(pnl_values),
        "pnl_values": pnl_values,
        "drawdowns": drawdowns,
    }


def run_monte_carlo(
    simulations: int,
    seed: int,
    historical_actuals: List[float],
    historical_errors: List[float],
    mean_error: float,
    error_sd: float,
    threshold: float,
    base_rate: float,
    shrinkage_lambda: float,
    market_reaction_speed: float,
    market_error_multiplier: float,
    market_noise_sd: float,
    half_spread: float,
    kelly_fraction: float,
    market_mode: str = "provided",
    efficiency_reaction_speed: float = 1.0,
    efficiency_noise_sd: float = 0.0,
) -> Dict:
    rng = random.Random(seed)

    pnl_values: List[float] = []
    drawdowns: List[float] = []
    transaction_counts: List[int] = []
    cap_hits = 0

    for _ in range(simulations):
        scenario = generate_scenario(
            historical_actuals,
            historical_errors,
            mean_error,
            error_sd,
            threshold,
            base_rate,
            rng,
            market_reaction_speed,
            market_error_multiplier,
            market_noise_sd,
            half_spread,
        )

        result = run_settlement_trading_day(
            scenario["updates"],
            scenario["actual_outcome"],
            base_rate,
            historical_errors,
            threshold,
            shrinkage_lambda,
            revision_weight=REVISION_WEIGHT,
            edge_threshold=EDGE_THRESHOLD,
            max_position=POSITION_CAP,
            starting_cash=STARTING_CASH,
            exit_buffer=EXIT_BUFFER,
            kelly_fraction=kelly_fraction,
            market_mode=market_mode,
            control_half_spread=half_spread,
            efficiency_reaction_speed=efficiency_reaction_speed,
            efficiency_noise_sd=efficiency_noise_sd,
        )

        pnl_values.append(result["final_pnl"])
        drawdowns.append(result["max_drawdown"])
        transaction_counts.append(result["transaction_count"])

        if result["hit_position_cap"]:
            cap_hits += 1

    return summarize_monte_carlo(
        pnl_values,
        drawdowns,
        transaction_counts,
        cap_hits,
    )


# ==================================================
# STRESS TESTS
# ==================================================


def run_market_efficiency_stress_test(
    simulations: int,
    historical_actuals: List[float],
    historical_errors: List[float],
    mean_error: float,
    error_sd: float,
    threshold: float,
    base_rate: float,
    shrinkage_lambda: float,
) -> List[Dict]:
    rows = []

    for noise_sd in MARKET_NOISE_GRID:
        for reaction_speed in REACTION_SPEED_GRID:
            summary = run_monte_carlo(
                simulations=simulations,
                seed=RANDOM_SEED,
                historical_actuals=historical_actuals,
                historical_errors=historical_errors,
                mean_error=mean_error,
                error_sd=error_sd,
                threshold=threshold,
                base_rate=base_rate,
                shrinkage_lambda=shrinkage_lambda,
                market_reaction_speed=BASELINE_MARKET_REACTION_SPEED,
                market_error_multiplier=BASELINE_MARKET_ERROR_MULTIPLIER,
                market_noise_sd=BASELINE_MARKET_NOISE_SD,
                half_spread=BASELINE_HALF_SPREAD,
                kelly_fraction=KELLY_FRACTION,
                market_mode="fair_value_convergence",
                efficiency_reaction_speed=reaction_speed,
                efficiency_noise_sd=noise_sd,
            )

            rows.append(
                {
                    "efficiency_noise_sd": noise_sd,
                    "market_reaction_speed": reaction_speed,
                    "mean_pnl": summary["mean_pnl"],
                    "median_pnl": summary["median_pnl"],
                    "pnl_sd": summary["pnl_sd"],
                    "win_rate": summary["win_rate"],
                    "p05": summary["p05"],
                    "p95": summary["p95"],
                    "average_drawdown": summary["average_drawdown"],
                    "cap_hit_rate": summary["cap_hit_rate"],
                }
            )

    return rows


def run_kelly_sensitivity(
    simulations: int,
    historical_actuals: List[float],
    historical_errors: List[float],
    mean_error: float,
    error_sd: float,
    threshold: float,
    base_rate: float,
    shrinkage_lambda: float,
) -> List[Dict]:
    rows = []

    for kelly_fraction in KELLY_FRACTION_GRID:
        summary = run_monte_carlo(
            simulations=simulations,
            seed=RANDOM_SEED,
            historical_actuals=historical_actuals,
            historical_errors=historical_errors,
            mean_error=mean_error,
            error_sd=error_sd,
            threshold=threshold,
            base_rate=base_rate,
            shrinkage_lambda=shrinkage_lambda,
            market_reaction_speed=BASELINE_MARKET_REACTION_SPEED,
            market_error_multiplier=BASELINE_MARKET_ERROR_MULTIPLIER,
            market_noise_sd=BASELINE_MARKET_NOISE_SD,
            half_spread=BASELINE_HALF_SPREAD,
            kelly_fraction=kelly_fraction,
            market_mode="provided",
        )

        rows.append(
            {
                "kelly_fraction": kelly_fraction,
                "mean_pnl": summary["mean_pnl"],
                "median_pnl": summary["median_pnl"],
                "pnl_sd": summary["pnl_sd"],
                "win_rate": summary["win_rate"],
                "p05": summary["p05"],
                "p95": summary["p95"],
                "worst": summary["worst"],
                "average_drawdown": summary["average_drawdown"],
                "p95_drawdown": summary["p95_drawdown"],
                "cap_hit_rate": summary["cap_hit_rate"],
            }
        )

    return rows


# ==================================================
# OUTPUT FILES
# ==================================================


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_validation_outputs(validation: Dict) -> None:
    write_csv(
        RESULTS_DIR / "walk_forward_predictions.csv",
        validation["rows"],
    )

    write_csv(
        RESULTS_DIR / "calibration_table.csv",
        validation["calibration_rows"],
    )


def save_stress_outputs(
    efficiency_rows: List[Dict],
    kelly_rows: List[Dict],
) -> None:
    write_csv(
        RESULTS_DIR / "market_efficiency_stress_test.csv",
        efficiency_rows,
    )

    write_csv(
        RESULTS_DIR / "kelly_sensitivity.csv",
        kelly_rows,
    )


# ==================================================
# CHARTS
# ==================================================


def create_charts(
    validation: Dict,
    baseline_monte_carlo: Dict,
    efficiency_rows: List[Dict],
    kelly_rows: List[Dict],
) -> None:
    if plt is None:
        print("\nmatplotlib is not installed; skipping charts.")
        return

    # --------------------------------------------------
    # 1. BRIER SCORE COMPARISON
    # --------------------------------------------------

    labels = ["Base rate", "Raw empirical", "Calibrated empirical"]
    values = [
        validation["baseline_brier"],
        validation["raw_brier"],
        validation["calibrated_brier"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Out-of-Sample Probability Validation")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "brier_score_comparison.png", dpi=180)
    plt.close(fig)

    # --------------------------------------------------
    # 2. CALIBRATION CURVE
    # --------------------------------------------------

    calibration_rows = validation["calibration_rows"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    ax.plot(
        [row["average_prediction"] for row in calibration_rows],
        [row["observed_rate"] for row in calibration_rows],
        marker="o",
        label="Calibrated empirical model",
    )

    ax.set_xlabel("Average predicted probability")
    ax.set_ylabel("Observed event frequency")
    ax.set_title("Walk-Forward Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "calibration_curve.png", dpi=180)
    plt.close(fig)

    # --------------------------------------------------
    # 3. MONTE CARLO P&L DISTRIBUTION
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(baseline_monte_carlo["pnl_values"], bins=35)
    ax.axvline(0.0, linestyle="--")
    ax.set_xlabel("Final P&L ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Baseline Informed-Trader Monte Carlo P&L")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pnl_distribution.png", dpi=180)
    plt.close(fig)

    # --------------------------------------------------
    # 4. MARKET EFFICIENCY HEATMAP
    # --------------------------------------------------

    matrix = []

    for noise_sd in MARKET_NOISE_GRID:
        row = []

        for reaction_speed in REACTION_SPEED_GRID:
            match = next(
                item
                for item in efficiency_rows
                if item["efficiency_noise_sd"] == noise_sd
                and item["market_reaction_speed"] == reaction_speed
            )
            row.append(match["mean_pnl"])

        matrix.append(row)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(REACTION_SPEED_GRID)))
    ax.set_xticklabels(REACTION_SPEED_GRID)
    ax.set_yticks(range(len(MARKET_NOISE_GRID)))
    ax.set_yticklabels(MARKET_NOISE_GRID)
    ax.set_xlabel("Market reaction speed")
    ax.set_ylabel("Market noise SD")
    ax.set_title("Mean P&L as Market Incorporates Fair Value")
    fig.colorbar(image, ax=ax, label="Mean P&L ($)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "market_efficiency_heatmap.png", dpi=180)
    plt.close(fig)

    # --------------------------------------------------
    # 5. KELLY SENSITIVITY
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [row["kelly_fraction"] for row in kelly_rows],
        [row["mean_pnl"] for row in kelly_rows],
        marker="o",
        label="Mean P&L",
    )
    ax.plot(
        [row["kelly_fraction"] for row in kelly_rows],
        [row["pnl_sd"] for row in kelly_rows],
        marker="o",
        label="P&L SD",
    )
    ax.set_xlabel("Kelly fraction")
    ax.set_ylabel("Dollars")
    ax.set_title("Sizing Sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "kelly_sensitivity.png", dpi=180)
    plt.close(fig)


# ==================================================
# PRINT HELPERS
# ==================================================


def print_monte_carlo_summary(title: str, summary: Dict) -> None:
    print("\n==========================================")
    print(title)
    print("==========================================")
    print("Simulations:", summary["simulations"])
    print("Mean P&L:", round(summary["mean_pnl"], 2))
    print("Median P&L:", round(summary["median_pnl"], 2))
    print("P&L SD:", round(summary["pnl_sd"], 2))
    print("Win rate:", f"{100 * summary['win_rate']:.1f}%")
    print("Loss rate:", f"{100 * summary['loss_rate']:.1f}%")
    print("Flat rate:", f"{100 * summary['flat_rate']:.1f}%")
    print("5th percentile:", round(summary["p05"], 2))
    print("95th percentile:", round(summary["p95"], 2))
    print("Worst:", round(summary["worst"], 2))
    print("Best:", round(summary["best"], 2))
    print("Average max drawdown:", round(summary["average_drawdown"], 2))
    print("95th percentile drawdown:", round(summary["p95_drawdown"], 2))
    print("Average transactions:", round(summary["average_transactions"], 2))
    print("Position-cap hit rate:", f"{100 * summary['cap_hit_rate']:.1f}%")


def print_efficiency_stress_test(rows: List[Dict]) -> None:
    print("\n==========================================")
    print("MARKET EFFICIENCY STRESS TEST")
    print("==========================================")
    print(
        "noise_sd  reaction  mean_pnl  median  pnl_sd  win_rate  p05  drawdown"
    )

    for row in rows:
        print(
            f"{row['efficiency_noise_sd']:<8.3f}  "
            f"{row['market_reaction_speed']:<8.2f}  "
            f"{row['mean_pnl']:<8.2f}  "
            f"{row['median_pnl']:<6.2f}  "
            f"{row['pnl_sd']:<6.2f}  "
            f"{100 * row['win_rate']:<7.1f}%  "
            f"{row['p05']:<6.2f}  "
            f"{row['average_drawdown']:.2f}"
        )


def print_kelly_sensitivity(rows: List[Dict]) -> None:
    print("\n==========================================")
    print("KELLY SIZING SENSITIVITY")
    print("==========================================")
    print("kelly  mean_pnl  pnl_sd  win_rate  p05  worst  avg_dd  cap_hit")

    for row in rows:
        print(
            f"{row['kelly_fraction']:<5.2f}  "
            f"{row['mean_pnl']:<8.2f}  "
            f"{row['pnl_sd']:<6.2f}  "
            f"{100 * row['win_rate']:<7.1f}%  "
            f"{row['p05']:<6.2f}  "
            f"{row['worst']:<6.2f}  "
            f"{row['average_drawdown']:<6.2f}  "
            f"{100 * row['cap_hit_rate']:.1f}%"
        )


# ==================================================
# MAIN
# ==================================================


def main() -> None:
    ensure_results_directory()

    # --------------------------------------------------
    # 1. LOAD + ALIGN HISTORICAL DATA
    # --------------------------------------------------

    forecast_dates, raw_forecasts = get_historical_forecasts(
        HISTORICAL_START_DATE,
        HISTORICAL_END_DATE,
    )

    actual_dates, raw_actuals = get_historical_actuals(
        HISTORICAL_START_DATE,
        HISTORICAL_END_DATE,
    )

    dates, forecasts, actuals = align_historical_data(
        forecast_dates,
        raw_forecasts,
        actual_dates,
        raw_actuals,
    )

    print("Aligned historical observations:", len(dates))

    # --------------------------------------------------
    # 2. HISTORICAL MODEL
    # --------------------------------------------------

    base_rate, yes_days, no_days = calculate_event_base_rate(
        actuals,
        RAIN_THRESHOLD,
    )

    mean_error, error_sd, historical_errors = analyze_forecast_errors(
        forecasts,
        actuals,
    )

    print("\n==========================================")
    print("HISTORICAL MODEL")
    print("==========================================")
    print("YES days:", yes_days)
    print("NO days:", no_days)
    print("Base rate:", round(base_rate, 4))
    print("Mean forecast error:", round(mean_error, 4))
    print("Forecast error SD:", round(error_sd, 4))

    # --------------------------------------------------
    # 3. NESTED WALK-FORWARD VALIDATION
    # --------------------------------------------------

    validation = run_calibrated_walk_forward_validation(
        dates,
        forecasts,
        actuals,
        RAIN_THRESHOLD,
        OUTER_VALIDATION_TRAINING_DAYS,
    )

    print("\n==========================================")
    print("OUT-OF-SAMPLE MODEL VALIDATION")
    print("==========================================")
    print("Predictions:", validation["predictions"])
    print("YES days:", validation["yes_days"])
    print("NO days:", validation["no_days"])
    print("Base-rate Brier:", round(validation["baseline_brier"], 4))
    print("Raw empirical Brier:", round(validation["raw_brier"], 4))
    print("Calibrated Brier:", round(validation["calibrated_brier"], 4))
    print(
        "Calibrated Brier skill:",
        f"{100 * validation['calibrated_skill']:.1f}%",
    )
    print("Calibration ECE:", round(validation["calibrated_ece"], 4))
    print("Average selected lambda:", round(validation["average_lambda"], 3))

    save_validation_outputs(validation)

    # --------------------------------------------------
    # 4. FINAL CALIBRATION PARAMETER
    # --------------------------------------------------

    shrinkage_lambda, lambda_scores = choose_shrinkage_lambda(
        forecasts,
        actuals,
        RAIN_THRESHOLD,
        INNER_CALIBRATION_TRAINING_DAYS,
    )

    print("\n==========================================")
    print("FINAL CALIBRATION PARAMETER")
    print("==========================================")
    print("Selected lambda:", shrinkage_lambda)
    print("Candidate lambda Brier scores:")

    for candidate_lambda in sorted(lambda_scores):
        print(
            f"  {candidate_lambda:.1f} -> "
            f"{lambda_scores[candidate_lambda]:.4f}"
        )

    # --------------------------------------------------
    # 5. LIVE WEATHER SNAPSHOT
    # --------------------------------------------------

    try:
        tomorrow, live_forecast = get_tomorrow_forecast()

        raw_live_probability = empirical_rainfall_probability(
            live_forecast,
            historical_errors,
            RAIN_THRESHOLD,
        )

        calibrated_live_probability = shrink_probability(
            raw_live_probability,
            base_rate,
            shrinkage_lambda,
        )

        print("\n==========================================")
        print("LIVE WEATHER SNAPSHOT")
        print("==========================================")
        print("Tomorrow:", tomorrow)
        print("Rainfall forecast:", round(live_forecast, 4))
        print("Raw empirical probability:", round(raw_live_probability, 4))
        print("Calibrated probability:", round(calibrated_live_probability, 4))

    except Exception as exc:
        print("\nLIVE WEATHER SNAPSHOT FAILED:", exc)
        print("Historical validation and simulations will continue.")

    # --------------------------------------------------
    # 6. BASELINE PROFIT-ORIENTED SYNTHETIC MONTE CARLO
    # --------------------------------------------------

    baseline_monte_carlo = run_monte_carlo(
        simulations=MONTE_CARLO_SIMULATIONS,
        seed=RANDOM_SEED,
        historical_actuals=actuals,
        historical_errors=historical_errors,
        mean_error=mean_error,
        error_sd=error_sd,
        threshold=RAIN_THRESHOLD,
        base_rate=base_rate,
        shrinkage_lambda=shrinkage_lambda,
        market_reaction_speed=BASELINE_MARKET_REACTION_SPEED,
        market_error_multiplier=BASELINE_MARKET_ERROR_MULTIPLIER,
        market_noise_sd=BASELINE_MARKET_NOISE_SD,
        half_spread=BASELINE_HALF_SPREAD,
        kelly_fraction=KELLY_FRACTION,
        market_mode="provided",
    )

    print_monte_carlo_summary(
        "BASELINE INFORMED-TRADER MONTE CARLO",
        baseline_monte_carlo,
    )

    # Save raw P&L distribution.
    write_csv(
        RESULTS_DIR / "baseline_monte_carlo_pnl.csv",
        [
            {"simulation": index + 1, "final_pnl": pnl}
            for index, pnl in enumerate(baseline_monte_carlo["pnl_values"])
        ],
    )

    # --------------------------------------------------
    # 7. ZERO-EDGE CONTROL
    # --------------------------------------------------

    zero_edge_control = run_monte_carlo(
        simulations=MONTE_CARLO_SIMULATIONS,
        seed=RANDOM_SEED,
        historical_actuals=actuals,
        historical_errors=historical_errors,
        mean_error=mean_error,
        error_sd=error_sd,
        threshold=RAIN_THRESHOLD,
        base_rate=base_rate,
        shrinkage_lambda=shrinkage_lambda,
        market_reaction_speed=BASELINE_MARKET_REACTION_SPEED,
        market_error_multiplier=BASELINE_MARKET_ERROR_MULTIPLIER,
        market_noise_sd=BASELINE_MARKET_NOISE_SD,
        half_spread=BASELINE_HALF_SPREAD,
        kelly_fraction=KELLY_FRACTION,
        market_mode="efficient_control",
    )

    print_monte_carlo_summary(
        "ZERO-EDGE CONTROL",
        zero_edge_control,
    )

    if (
        abs(zero_edge_control["mean_pnl"]) < 1e-9
        and zero_edge_control["average_transactions"] == 0
        and abs(zero_edge_control["flat_rate"] - 1.0) < 1e-9
    ):
        print("CONTROL TEST: PASS")
    else:
        print("CONTROL TEST: WARNING - inspect strategy logic.")

    # --------------------------------------------------
    # 8. MARKET-EFFICIENCY STRESS TEST
    # --------------------------------------------------

    efficiency_rows = run_market_efficiency_stress_test(
        simulations=MONTE_CARLO_SIMULATIONS,
        historical_actuals=actuals,
        historical_errors=historical_errors,
        mean_error=mean_error,
        error_sd=error_sd,
        threshold=RAIN_THRESHOLD,
        base_rate=base_rate,
        shrinkage_lambda=shrinkage_lambda,
    )

    print_efficiency_stress_test(efficiency_rows)

    # --------------------------------------------------
    # 9. KELLY SIZING SENSITIVITY
    # --------------------------------------------------

    kelly_rows = run_kelly_sensitivity(
        simulations=MONTE_CARLO_SIMULATIONS,
        historical_actuals=actuals,
        historical_errors=historical_errors,
        mean_error=mean_error,
        error_sd=error_sd,
        threshold=RAIN_THRESHOLD,
        base_rate=base_rate,
        shrinkage_lambda=shrinkage_lambda,
    )

    print_kelly_sensitivity(kelly_rows)

    save_stress_outputs(efficiency_rows, kelly_rows)

    # --------------------------------------------------
    # 10. CHARTS
    # --------------------------------------------------

    create_charts(
        validation,
        baseline_monte_carlo,
        efficiency_rows,
        kelly_rows,
    )

    # --------------------------------------------------
    # 11. FINAL PROJECT SUMMARY
    # --------------------------------------------------

    print("\n==========================================")
    print("FINAL PROJECT SUMMARY")
    print("==========================================")
    print("Probability model: calibrated empirical forecast-error model")
    print("Calibration: nested walk-forward shrinkage")
    print("Selected lambda:", shrinkage_lambda)
    print("Revision weight:", REVISION_WEIGHT)
    print("Minimum executable edge:", EDGE_THRESHOLD)
    print("Sizing: fractional Kelly")
    print("Kelly fraction:", KELLY_FRACTION)
    print("Position cap:", POSITION_CAP)
    print("Settlement: $1 correct / $0 incorrect")
    print("Baseline market reaction speed:", BASELINE_MARKET_REACTION_SPEED)
    print("Baseline market error multiplier:", BASELINE_MARKET_ERROR_MULTIPLIER)
    print("Baseline total spread:", round(2 * BASELINE_HALF_SPREAD, 3))
    print("Efficiency stress test: market converges toward trader fair value")
    print("Monte Carlo paths:", MONTE_CARLO_SIMULATIONS)
    print(
        "OOS calibrated Brier:",
        round(validation["calibrated_brier"], 4),
    )
    print(
        "OOS Brier skill vs base rate:",
        f"{100 * validation['calibrated_skill']:.1f}%",
    )
    print(
        "Baseline simulated mean P&L:",
        round(baseline_monte_carlo["mean_pnl"], 2),
    )
    print(
        "Baseline simulated median P&L:",
        round(baseline_monte_carlo["median_pnl"], 2),
    )
    print(
        "Baseline simulated win rate:",
        f"{100 * baseline_monte_carlo['win_rate']:.1f}%",
    )
    print(
        "Zero-edge control mean P&L:",
        round(zero_edge_control["mean_pnl"], 2),
    )
    print("Results saved to:", RESULTS_DIR.resolve())

    print("\nIMPORTANT INTERPRETATION")
    print(
        "Positive Monte Carlo P&L is conditional on the synthetic-market "
        "assumption that the trader has a faster/cleaner weather signal than "
        "the market. It is not evidence of guaranteed real-world profit."
    )


if __name__ == "__main__":
    main()
