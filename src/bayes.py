def bayesian_update(
    prior,
    likelihood_if_event,
    likelihood_if_no_event
):
    numerator = (
        likelihood_if_event
        * prior
    )

    denominator = (
        numerator
        + likelihood_if_no_event
        * (1 - prior)
    )

    posterior = (
        numerator
        / denominator
    )

    return posterior