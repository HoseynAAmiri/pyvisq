import numpy as np

from functools import lru_cache
from dataclasses import dataclass

from .model import Model
from .elements import Spring, Dashpot, Springpot, SpringParams, DashpotParams, SpringpotParams
from ..utils import mittleff


@dataclass
class MaxwellParams():
    dashpot: DashpotParams
    spring: SpringParams


class Maxwell(Model):
    _params: MaxwellParams
    spring: Spring
    dashpot: Dashpot
    diagram = """
                ___
            _____| |_______╱╲  ╱╲  ╱╲  ___
                _|_|  c      ╲╱  ╲╱  ╲╱  k
            """

    def __init__(self, params: MaxwellParams) -> None:
        super().__init__()
        self.params = params
        self.dashpot = Dashpot(params.dashpot)
        self.spring = Spring(params.spring)

    @property
    def params(self) -> MaxwellParams:
        return self._params

    @params.setter
    def params(self, params: MaxwellParams) -> None:
        if not isinstance(params, MaxwellParams):
            raise ValueError("Invalid parameters: Expected MaxwellParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        c = self.params.dashpot.c
        k = self.params.spring.k
        tau = c / k
        return k * np.exp(-t / tau)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.dashpot.J(t) + self.spring.J(t)


@dataclass
class FracDashpotMaxwellParams():
    dashpot: DashpotParams
    springpot: SpringpotParams


class FracDashpotMaxwell(Model):
    _params: FracDashpotMaxwellParams
    dashpot: Dashpot
    springpot: Springpot
    diagram = """
                ___
            _____| |_______╱╲____
                _|_|  c    ╲╱  cb, b
            """

    def __init__(self, params: FracDashpotMaxwellParams) -> None:
        super().__init__()
        self.params = params
        self.dashpot = Dashpot(params.dashpot)
        self.springpot = Springpot(params.springpot)

    @property
    def params(self) -> FracDashpotMaxwellParams:
        return self._params

    @params.setter
    def params(self, params: FracDashpotMaxwellParams) -> None:
        if not isinstance(params, FracDashpotMaxwellParams):
            raise ValueError(
                "Invalid parameters: Expected FracDashpotMaxwellParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        c = self.params.dashpot.c
        b = self.params.springpot.e
        cb = self.params.springpot.ce
        return cb * t ** (-b) * mittleff(1 - b, 1 - b, -cb * t ** (1 - b) / c)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.dashpot.J(t) + self.springpot.J(t)


@dataclass
class FracSpringMaxwellParams():
    springpot: SpringpotParams
    spring: SpringParams


class FracSpringMaxwell(Model):
    _params: FracSpringMaxwellParams
    springpot: Springpot
    spring: Spring
    diagram = """
            ____╱╲_________╱╲  ╱╲  ╱╲  ______
                ╲╱  ca, a    ╲╱  ╲╱  ╲╱  k
            """

    def __init__(self, params: FracSpringMaxwellParams) -> None:
        super().__init__()
        self.params = params
        self.spring = Spring(params.spring)
        self.springpot = Springpot(params.springpot)

    @property
    def params(self) -> FracSpringMaxwellParams:
        return self._params

    @params.setter
    def params(self, params: FracSpringMaxwellParams) -> None:
        if not isinstance(params, FracSpringMaxwellParams):
            raise ValueError(
                "Invalid parameters: Expected FracSpringMaxwellParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        k = self.params.spring.k
        a = self.params.springpot.e
        ca = self.params.springpot.ce
        return k * mittleff(a, a, -k * t ** (a) / ca)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.spring.J(t) + self.springpot.J(t)


@dataclass
class FracMaxwellParams():
    springpot_a: SpringpotParams
    springpot_b: SpringpotParams


class FracMaxwell(Model):
    _params: FracMaxwellParams
    springpot_a: Springpot
    springpot_b: Springpot
    diagram = """
            ____╱╲___________╱╲______
                ╲╱  ca, a    ╲╱  cb, b
            """

    def __init__(self, params: FracMaxwellParams) -> None:
        super().__init__()
        self.params = params
        self.springpot_a = Springpot(params.springpot_a)
        self.springpot_b = Springpot(params.springpot_b)

    @property
    def params(self) -> FracMaxwellParams:
        return self._params

    @params.setter
    def params(self, params: FracMaxwellParams) -> None:
        if not isinstance(params, FracMaxwellParams):
            raise ValueError("Invalid parameters: Expected FracMaxwellParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        a = self.params.springpot_a.e
        ca = self.params.springpot_a.ce
        b = self.params.springpot_b.e
        cb = self.params.springpot_b.ce
        return cb * t ** (-b) * mittleff(a - b, 1 - b, -cb * t ** (a - b) / ca)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.springpot_a.J(t) + self.springpot_b.J(t)
