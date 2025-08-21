# issues:
they used GMM; though GMM ≈ OLS here because: (1) no instruments used, (2) Fed surprises are exogenous ?!

original results: −7.88, 4.08, 0.18

original y: 100log(price)

our y input: np.exp(-sp500_returns/100.0) ** 100