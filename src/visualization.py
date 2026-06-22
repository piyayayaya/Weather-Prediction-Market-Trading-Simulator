from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


def plot_trade_log(trade_log):
    time_steps = [row["time_step"] for row in trade_log]
    market_prices = [row["market_price"] for row in trade_log]
    model_probabilities = [row["model_probability"] for row in trade_log]
    yes_inventories = [row["yes_inventory"] for row in trade_log]
    pnls = [row["pnl"] for row in trade_log]

    plt.figure()
    plt.plot(time_steps, market_prices, marker="o")
    plt.plot(time_steps, model_probabilities, marker="o")
    plt.title("Market Price vs Model Probability")
    plt.xlabel("Time Step")
    plt.ylabel("Probability / Price")
    plt.legend(["Market Price", "Model Probability"])
    plt.savefig(FIGURES_DIR / "market_price_vs_model_probability.png")
    plt.show()

    plt.figure()
    plt.plot(time_steps, yes_inventories, marker="o")
    plt.title("YES Inventory Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("YES Inventory")
    plt.savefig(FIGURES_DIR / "yes_inventory_path.png")
    plt.show()

    plt.figure()
    plt.plot(time_steps, pnls, marker="o")
    plt.title("PnL Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("PnL")
    plt.savefig(FIGURES_DIR / "pnl_path.png")
    plt.show()


def plot_monte_carlo_pnl_distribution(results):
    final_pnls = [
        row["final_pnl"]
        for row in results
    ]

    plt.figure()
    plt.hist(final_pnls, bins=20)
    plt.title("Monte Carlo Final PnL Distribution")
    plt.xlabel("Final PnL")
    plt.ylabel("Frequency")
    plt.savefig(FIGURES_DIR / "monte_carlo_pnl_distribution.png")
    plt.show()