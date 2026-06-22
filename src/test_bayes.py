from bayes import bayesian_update

posterior = bayesian_update(
    prior=0.70,
    likelihood_if_event=0.80,
    likelihood_if_no_event=0.30
)

print(
    "Posterior Probability:",
    round(posterior, 4)
)