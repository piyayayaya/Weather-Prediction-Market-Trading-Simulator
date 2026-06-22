def generate_summary_statistics(
    trade_log
):
    total_trades = sum(
        row["trade_executed"]
        for row in trade_log
    )

    blocked_trades = sum(
        row["trade_blocked_by_risk_limit"]
        for row in trade_log
    )

    max_inventory = max(
        abs(row["yes_inventory"])
        for row in trade_log
    )

    average_edge = (
        sum(
            abs(row["edge"])
            for row in trade_log
        )
        / len(trade_log)
    )

    final_pnl = trade_log[-1]["pnl"]

    return {
        "Total Trades": total_trades,
        "Blocked Trades": blocked_trades,
        "Max Inventory": max_inventory,
        "Average Edge": round(
            average_edge,
            4
        ),
        "Final PnL": round(
            final_pnl,
            2
        )
    }