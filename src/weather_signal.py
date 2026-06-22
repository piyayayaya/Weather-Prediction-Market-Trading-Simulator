def get_signal_likelihoods(
    update_type
):
    if update_type == "bullish_update":
        return (
            0.80,
            0.30
        )

    elif update_type == "bearish_update":
        return (
            0.30,
            0.80
        )

    else:
        return (
            0.55,
            0.45
        )