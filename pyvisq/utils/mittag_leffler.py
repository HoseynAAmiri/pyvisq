import warnings

import numpy as np

from scipy.special import gamma, gammaln

def mittleff(
    a: float,
    b: float,
    z: float,
    max_terms: int = 1000,
    tol: float = 1e-5
) -> float:
    result = 0.0
    abs_z = abs(z)
    if abs_z == 0:
        if b > 0:
            return 1.0 / gamma(b)
        else:
            warnings.warn("Invalid parameters: log(0) encountered.", RuntimeWarning)
            return float('nan')
    sign_z = np.sign(z)
    for n in range(max_terms):
        log_term = n * np.log(abs_z) - gammaln(a * n + b)
        term = np.exp(log_term) * sign_z**n
        result += term
        if abs(term) < tol * abs(result):
            break

    if n == max_terms - 1:
        warnings.warn(
            f"_ml failed to converge with {n} terms. Increase max_terms.", RuntimeWarning
            )

    return result
