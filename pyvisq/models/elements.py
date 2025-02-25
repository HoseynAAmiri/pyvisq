import numpy as np

from scipy.special import gamma
from functools import lru_cache
from dataclasses import dataclass

from .model import Model


@dataclass
class SpringParams():
    k: float

    def __post_init__(self):
        if self.k <= 0:
            raise ValueError("k must be greater than zero.")


class Spring(Model):
    _params: SpringParams
    diagram = """
                ____╱╲  ╱╲  ╱╲  ____
                      ╲╱  ╲╱  ╲╱  k
                """

    def __init__(self, params: SpringParams) -> None:
        super().__init__()
        self.params = params

    @property
    def params(self) -> SpringParams:
        return self._params

    @params.setter
    def params(self, params: SpringParams) -> None:
        if not isinstance(params, SpringParams):
            raise ValueError("Invalid parameters: Expected SpringParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        k = self.params.k
        return k

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        k = self.params.k
        return 1 / k


@dataclass
class DashpotParams():
    c: float

    def __post_init__(self):
        if self.c <= 0:
            raise ValueError("c must be greater than zero.")


class Dashpot(Model):
    _params: DashpotParams
    diagram = """
                    ___
                _____| |_____
                    _|_|  c
                """

    def __init__(self, params: DashpotParams) -> None:
        super().__init__()
        self.params = params

    @property
    def params(self) -> DashpotParams:
        return self._params

    @params.setter
    def params(self, params: DashpotParams) -> None:
        if not isinstance(params, DashpotParams):
            raise ValueError("Invalid parameters: Expected DashpotParams.")
        self._params = params

    def G(self, t: float) -> float:
        dirac = np.inf if t == 0 else 0
        return self.params.c * dirac

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        c = self.params.c
        return t / c


@dataclass
class SpringpotParams():
    e: float
    ce: float

    def __post_init__(self):
        if not (0 <= self.e <= 1):
            raise ValueError("b must be between 0 and 1.")
        if self.ce <= 0:
            raise ValueError("cb must be greater than zero.")


class Springpot(Model):
    _params: SpringpotParams
    diagram = """
                ____╱╲____
                    ╲╱  ce, e
                """

    def __init__(self, params: SpringpotParams) -> None:
        super().__init__()
        self.params = params

    @property
    def params(self) -> SpringpotParams:
        return self._params

    @params.setter
    def params(self, params: SpringpotParams) -> None:
        if not isinstance(params, SpringpotParams):
            raise ValueError("Invalid parameters: Expected SpringpotParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        t = max(t, 1e-10)
        e = self.params.e
        ce = self.params.ce
        return ce * t ** (-e) / gamma(1 - e)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        e = self.params.e
        ce = self.params.ce
        return (t ** e) / (ce * gamma(1 + e))
