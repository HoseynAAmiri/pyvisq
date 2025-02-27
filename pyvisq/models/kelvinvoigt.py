import numpy as np

from functools import lru_cache
from dataclasses import dataclass

from .model import Model
from .elements import Spring, Dashpot, Springpot, SpringParams, DashpotParams, SpringpotParams
from ..utils import ml


@dataclass
class KelvinVoigtParams():
    dashpot: DashpotParams
    spring: SpringParams


class KelvinVoigt(Model):
    _params: KelvinVoigtParams
    dashpot: Dashpot
    spring: Spring
    diagram = """
                        ___
                 ________| |_________
                |       _|_|  c      |
            ____|                    |____
                |                    |
                |____╱╲  ╱╲  ╱╲  ____|
                       ╲╱  ╲╱  ╲╱  k
            """

    def __init__(self, params: KelvinVoigtParams) -> None:
        super().__init__()
        self.params = params
        self.spring = Spring(params.spring)
        self.dashpot = Dashpot(params.dashpot)

    @property
    def params(self) -> KelvinVoigtParams:
        return self._params

    @params.setter
    def params(self, params: KelvinVoigtParams) -> None:
        if not isinstance(params, KelvinVoigtParams):
            raise ValueError("Invalid parameters: Expected KelvinVoigtParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.spring.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        k = self.params.spring.k
        c = self.params.dashpot.c
        tau = c / k
        return (1 - np.exp(-t / tau)) / k


@dataclass
class FracDashpotKelvinVoigtParams():
    dashpot: DashpotParams
    springpot: SpringpotParams


class FracDashpotKelvinVoigt(Model):
    _params: FracDashpotKelvinVoigtParams
    dashpot: Dashpot
    springpot: Springpot
    diagram = """
                        ___
                 ________| |_________
                |       _|_|  c      |
            ____|                    |____
                |                    |
                |_________╱╲_________|
                          ╲╱  cb, b
            """

    def __init__(self, params: FracDashpotKelvinVoigtParams) -> None:
        super().__init__()
        self.params = params
        self.dashpot = Dashpot(params.dashpot)
        self.springpot = Springpot(params.springpot)

    @property
    def params(self) -> FracDashpotKelvinVoigtParams:
        return self._params

    @params.setter
    def params(self, params: FracDashpotKelvinVoigtParams) -> None:
        if not isinstance(params, FracDashpotKelvinVoigtParams):
            raise ValueError(
                "Invalid parameters: Expected FracDashpotKelvinVoigtParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.springpot.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        t = max(t, 1e-10)
        c = self.params.dashpot.c
        b = self.params.springpot.e
        cb = self.params.springpot.ce
        return (t / c) * ml(1 - b, 1 + 1, -cb * t ** (1 - b) / c)


@dataclass
class FracSpringKelvinVoigtParams():
    springpot: SpringpotParams
    spring: SpringParams


class FracSpringKelvinVoigt(Model):
    _params: FracSpringKelvinVoigtParams
    springpot: Springpot
    spring: Spring
    diagram = """
                 _________╱╲_________
                |         ╲╱  ca, a  |
            ____|                    |____
                |                    |
                |_____╱╲  ╱╲  ╱╲  ___|
                        ╲╱  ╲╱  ╲╱  k
            """

    def __init__(self, params: FracSpringKelvinVoigtParams) -> None:
        super().__init__()
        self.params = params
        self.spring = Spring(params.spring)
        self.springpot = Springpot(params.springpot)

    @property
    def params(self) -> FracSpringKelvinVoigtParams:
        return self._params

    @params.setter
    def params(self, params: FracSpringKelvinVoigtParams) -> None:
        if not isinstance(params, FracSpringKelvinVoigtParams):
            raise ValueError(
                "Invalid parameters: Expected FracSpringKelvinVoigtParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.spring.G(t) + self.springpot.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        t = max(t, 1e-10)
        k = self.params.spring.k
        a = self.params.springpot.e
        ca = self.params.springpot.ce
        return (t ** a / ca) * ml(a, 1 + a, -k * t ** a / ca)


@dataclass
class FracKelvinVoigtParams():
    springpot_a: SpringpotParams
    springpot_b: SpringpotParams


class FracKelvinVoigt(Model):
    _params: FracKelvinVoigtParams
    springpot_a: Springpot
    springpot_b: Springpot
    diagram = """
                 _________╱╲_________
                |         ╲╱  ca, a  |
            ____|                    |____
                |                    |
                |_________╱╲_________|
                          ╲╱  cb, b
            """

    def __init__(self, params: FracKelvinVoigtParams) -> None:
        super().__init__()
        self.params = params
        self.springpot_a = Springpot(params.springpot_a)
        self.springpot_b = Springpot(params.springpot_b)

    @property
    def params(self) -> FracKelvinVoigtParams:
        return self._params

    @params.setter
    def params(self, params: FracKelvinVoigtParams) -> None:
        if not isinstance(params, FracKelvinVoigtParams):
            raise ValueError(
                "Invalid parameters: Expected FracKelvinVoigtParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.springpot_a.G(t) + self.springpot_b.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        t = max(t, 1e-10)
        a = self.params.springpot_a.e
        ca = self.params.springpot_a.ce
        b = self.params.springpot_b.e
        cb = self.params.springpot_b.ce
        return (t ** a / ca) * ml(a - b, 1 + a, -cb * t ** (a - b) / ca)
