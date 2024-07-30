import numpy as np

import numpy.typing as npt

from scipy.special import gamma

from .model import Model
from .utils.mittag_leffler import ml as ml_accurate


def ml_fast(z: npt.NDArray[np.float64], a: float, b: float, inf: int = 100) -> npt.NDArray[np.float64]:
    z = np.asarray(z)
    result = np.zeros_like(z)
    for n in range(inf):
        term = (z ** n) / gamma(a * n + b)
        result += term
    return result


def ml_vec(z: npt.NDArray[np.float64], alpha: float, beta: float, inf: int = 100):
    k = np.arange(0, inf, dtype=int)[:, np.newaxis]
    return (z**k/gamma(alpha*k + beta)).sum(axis=0)


class FractionalSLS(Model):
    mode: str
    A: float
    E1: float
    T1: float
    CA: float
    B: float
    E2: float
    T2: float
    CB: float
    EK: float

    E: float
    T: float

    def __init__(self, config: dict[str, float], mode='fast') -> None:
        super().__init__()
        self.mode = mode
        if all(k in config for k in ['A', 'CA', 'B', 'CB', 'EK']):
            self._initialize_direct(config)
        elif all(k in config for k in ['A', 'E1', 'T1', 'E2', 'T2', 'B', 'EK']):
            self._initialize_computed(config)
        else:
            raise ValueError(
                "Invalid configuration: Provide either (A, CA, B, CB, EK) or (A, E1, T1, B, E2, T2, EK)")

    def _initialize_direct(self, config: dict[str, float]) -> None:
        required_attrs = ['A', 'CA', 'B', 'CB', 'EK']
        for attr in required_attrs:
            setattr(self, attr, config[attr])

    def _initialize_computed(self, config: dict[str, float]) -> None:
        required_attrs = ['A', 'E1', 'T1', 'E2', 'T2', 'B', 'EK']
        for attr in required_attrs:
            setattr(self, attr, config[attr])

    def make(self) -> None:
        if all(hasattr(self, attr) for attr in ['E1', 'T1', 'E2', 'T2']):
            self.CA = self.E1 * self.T1 ** self.A
            self.CB = self.E2 * self.T2 ** self.B

        self.T = (self.CA / self.CB) ** (1 / (self.A - self.B))
        self.E = self.CA / (self.T ** self.A)

    def run(self, dwell_size: int = 1001) -> None:
        self.make()
        self.time = self._get_time(self.D, self.L, dwell_size)
        u = np.heaviside
        self.strain = (self.I * self.time / self.D) * (1 - u(self.time - self.D, 0.5)) + \
            self.I * u(self.time - self.D, 0.5)
        modulus = np.zeros_like(self.time)

        if self.mode == 'fast':
            func = ml_fast
        elif self.mode == 'vectorized':
            func = ml_vec
        else:
            func = ml_accurate

        modulus[1:] = self.E * (self.time[1:] / self.T) ** (-self.B) * \
            func(-(self.time[1:] / self.T)
                 ** (self.A - self.B), self.A-self.B, 1-self.B) + self.EK

        self.stress = modulus * self.strain
