import warnings

import numpy as np

from pymittagleffler import mittag_leffler
from scipy.special import gamma, gammaln


def ml(a: float, b: float, z: float) -> float:
    return np.real(mittag_leffler(z, a, b))


class MittagLeffler:
    # sloppy implementation of Mittag-Leffler function
    last_z: float
    z_thresh: float | None
    thresh_eval: float

    def __init__(self) -> None:
        self.last_z = -np.inf
        self.z_thresh = None
        self.thresh_eval = np.inf

    def __call__(self, a: float, b: float, z: float) -> float:
        if self.z_thresh is None:
            eval_current = mittleff(a, b, z)
            z = abs(z)
            if z > self.last_z:
                self.last_z = z
                if abs(eval_current) <= self.thresh_eval:
                    self.thresh_eval = eval_current
                    return eval_current
                else:
                    self.z_thresh = z
                    return self.thresh_eval
            else:
                return eval_current
        else:
            if abs(z) < self.z_thresh:
                return mittleff(a, b, z)
            else:
                return self.thresh_eval


def mittleff(
    a: float,
    b: float,
    z: float,
    max_terms: int = 1000,
    tol: float = 1e-5
) -> float:
    result = 0.0
    abs_z = abs(z)
    if abs_z < 1.e-15:
        if b > 0:
            return 1.0 / gamma(b)
        else:
            warnings.warn(
                "Invalid parameters: log(0) encountered.", RuntimeWarning
            )
            return np.nan
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
