from functools import lru_cache
from dataclasses import dataclass

from .model import Model


@dataclass
class PowerlawParams():
    modulus_0: float
    modulus_inf: float
    tau: float
    e: float

    def __post_init__(self):
        if self.modulus_0 <= 0:
            raise ValueError("modulus_0 must be greater than zero.")
        if self.modulus_inf <= 0:
            raise ValueError("modulus_inf must be greater than zero.")
        if self.tau <= 0:
            raise ValueError("tau must be greater than zero.")
        if not (0 <= self.e <= 1):
            raise ValueError("e must be between 0 and 1.")


class Powerlaw(Model):
    _params: PowerlawParams
    diagram = ""

    def __init__(self, params: PowerlawParams) -> None:
        super().__init__()
        self.params = params

    @property
    def params(self) -> PowerlawParams:
        return self._params

    @params.setter
    def params(self, params: PowerlawParams) -> None:
        if not isinstance(params, PowerlawParams):
            raise ValueError("Invalid parameters: Expected PowerlawParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.params.modulus_inf + (self.params.modulus_0 - self.params.modulus_inf) * (1 + t / self.params.tau) ** (-self.params.e)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return 1 / self.params.modulus_inf + (1 / self.params.modulus_0) * (t / self.params.tau) ** self.params.e
