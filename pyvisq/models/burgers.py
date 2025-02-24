import numpy as np

from functools import lru_cache
from dataclasses import dataclass

from .model import Model
from .maxwell import Maxwell, MaxwellParams
from .kelvinvoigt import KelvinVoigt, KelvinVoigtParams
from .elements import Spring, Dashpot, SpringParams, DashpotParams


@dataclass
class BurgersParams():
    dashpot_a: DashpotParams
    spring_a: SpringParams
    dashpot_b: DashpotParams
    spring_b: SpringParams


class Burgers(Model):
    _params: BurgersParams
    dashpot_a: Dashpot
    spring_a: Spring
    maxwell_branch: Maxwell
    dashpot_b: Dashpot
    spring_b: Spring
    kelvinvoigt_branch: KelvinVoigt
    diagram = """
                                                        ___
                                                _________| |________
                ___                            |        _|_|  cb    |
            _____| |________╱╲  ╱╲  ╱╲  _______|                    |____
                _|_|  ca      ╲╱  ╲╱  ╲╱  ka   |                    |
                                               |____╱╲  ╱╲  ╱╲  ____|
                                                      ╲╱  ╲╱  ╲╱  kb
            """

    def __init__(self, params: BurgersParams) -> None:
        super().__init__()
        self.params = params
        self.maxwell_branch = Maxwell(
            MaxwellParams(params.dashpot_a, params.spring_a)
        )
        self.dashpot_a = self.maxwell_branch.dashpot
        self.spring_a = self.maxwell_branch.spring
        self.kelvinvoigt_branch = KelvinVoigt(
            KelvinVoigtParams(params.dashpot_a, params.spring_a)
        )
        self.dashpot_b = self.kelvinvoigt_branch.dashpot
        self.spring_b = self.kelvinvoigt_branch.spring

    @property
    def params(self) -> BurgersParams:
        return self._params

    @params.setter
    def params(self, params: BurgersParams) -> None:
        if not isinstance(params, BurgersParams):
            raise ValueError("Invalid parameters: Expected PowerlawParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        ca = self.params.dashpot_a.c
        ka = self.params.spring_a.k
        cb = self.params.dashpot_b.c
        kb = self.params.spring_b.k
        p1 = ca/ka + ca/kb + cb/kb
        p2 = ca * cb / (ka * kb)
        q1 = ca
        q2 = ca * cb / kb
        A = np.sqrt(p1**2 - 4*p2)
        r1 = (p1 - A) / (2 * p2)
        r2 = (p1 + A) / (2 * p2)
        return ((q1 - q2 * r1) * np.exp(-r1 * t) - (q1 - q2 * r2) * np.exp(-r2 * t)) / A

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.spring_a.J(t) + self.dashpot_a.J(t) + self.kelvinvoigt_branch.J(t)
