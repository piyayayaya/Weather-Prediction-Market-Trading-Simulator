from statistics import mean, stdev
import csv
from pathlib import Path
from src.visualization import plot_monte_carlo_pnl_distribution

from src.backtest import run_monte_carlo_backtest


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(exist_ok=True)

results = run_monte_carlo_backtest(
    num_simulations=100
)

final_pnls = [
    row["final_pnl"]
    for row in results
]

average_final_pnl = mean(
    final_pnls
)

pnl_std = stdev(
    final_pnls
)

profitable_simulations = sum(
    pnl > 0
    for pnl in final_pnls
)

win_rate = profitable_simulations / len(
    final_pnls
)

if pnl_std > 0:
    sharpe_like_ratio = (
        average_final_pnl / pnl_std
    )
else:
    sharpe_like_ratio = 0

csv_path = DATA_DIR / "monte_carlo_results.csv"

with open(
    csv_path,
    mode="w",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print("Monte Carlo Backtest Results")
print("Number of Simulations:", len(results))
print("Average Final PnL:", round(average_final_pnl, 4))
print("PnL Standard Deviation:", round(pnl_std, 4))
print("Win Rate:", round(win_rate, 4))
print("Sharpe-Like Ratio:", round(sharpe_like_ratio, 4))
print("Best Final PnL:", max(final_pnls))
print("Worst Final PnL:", min(final_pnls))
print("Saved Results To:", csv_path)

plot_monte_carlo_pnl_distribution(
    results
    )