import mpmath

import numpy as np

from functools import lru_cache
from dataclasses import dataclass

from .model import Model
from .elements import Spring, Dashpot, Springpot, SpringParams, DashpotParams, SpringpotParams
from .maxwell import Maxwell, MaxwellParams, FracDashpotMaxwell, FracDashpotMaxwellParams, FracSpringMaxwell, FracSpringMaxwellParams, FracMaxwell, FracMaxwellParams


@dataclass
class SLSParams():
    dashpot_a: DashpotParams
    spring_b: SpringParams
    spring_c: SpringParams


class SLS(Model):
    _params: SLSParams
    dashpot_a: Dashpot
    spring_b: Spring
    maxwell_branch: Maxwell
    spring_c: Spring
    diagram = """
                    ___                 
                 ____| |______╱╲  ╱╲  ╱╲  _____
                |   _|_|  ca    ╲╱  ╲╱  ╲╱  kb |
            ____|                              |____
                |                              |
                |__________╱╲  ╱╲  ╱╲  ________|
                             ╲╱  ╲╱  ╲╱  kc
            """

    def __init__(self, params: SLSParams) -> None:
        super().__init__()
        self.params = params
        self.maxwell_branch = Maxwell(
            MaxwellParams(params.dashpot_a, params.spring_b)
        )
        self.dashpot_a = self.maxwell_branch.dashpot
        self.spring_b = self.maxwell_branch.spring
        self.spring_c = Spring(params.spring_c)

    @property
    def params(self) -> SLSParams:
        return self._params

    @params.setter
    def params(self, params: SLSParams) -> None:
        if not isinstance(params, SLSParams):
            raise ValueError("Invalid parameters: Expected SLSParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.spring_c.G(t) + self.maxwell_branch.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        ca = self.params.dashpot_a.c
        kb = self.params.spring_b.k
        kc = self.params.spring_c.k
        c0 = 1 / kc
        c1 = kb / (kc * (kc + kb))
        tau = ca * (kc + kb) / (kc * kb)
        return c0 - c1 * np.exp(-t / tau)


@dataclass
class JeffreysZenerParams():
    dashpot_a: DashpotParams
    spring_b: SpringParams
    dashpot_c: DashpotParams


class JeffreysZener(Model):
    _params: JeffreysZenerParams
    dashpot_a: Dashpot
    spring_b: Spring
    maxwell_branch: Maxwell
    dashpot_c: Dashpot
    diagram = """
                    ___                 
                 ____| |______╱╲  ╱╲  ╱╲  _____
                |   _|_|  ca    ╲╱  ╲╱  ╲╱  kb |
            ____|                              |____
                |              ___             |
                |_______________| |____________|
                               _|_|  cc
            """

    def __init__(self, params: JeffreysZenerParams) -> None:
        super().__init__()
        self.params = params
        self.maxwell_branch = Maxwell(
            MaxwellParams(params.dashpot_a, params.spring_b)
        )
        self.dashpot_a = self.maxwell_branch.dashpot
        self.spring_b = self.maxwell_branch.spring
        self.dashpot_c = Dashpot(params.dashpot_c)

    @property
    def params(self) -> JeffreysZenerParams:
        return self._params

    @params.setter
    def params(self, params: JeffreysZenerParams) -> None:
        if not isinstance(params, JeffreysZenerParams):
            raise ValueError(
                "Invalid parameters: Expected JeffreysZenerParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.dashpot_c.G(t) + self.maxwell_branch.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        if t == 0:
            return 0

        ca = self.params.dashpot_a.c
        kb = self.params.spring_b.k
        cc = self.params.dashpot_c.c

        def J_hat(s):
            numerator = ca * s + kb
            denominator = ca * kb * s + cc * s * (ca * s + kb)
            return (1 / s) * numerator / denominator

        t_mp = mpmath.mpf(t)
        J_t = mpmath.invertlaplace(J_hat, t_mp, method='talbot')
        return float(J_t)


@dataclass
class FracJeffreysZenerParams():
    dashpot_a: DashpotParams
    springpot_b: SpringpotParams
    dashpot_c: DashpotParams


class FracJeffreysZener(Model):
    _params: FracJeffreysZenerParams
    dashpot_a: Dashpot
    springpot_b: Springpot
    maxwell_branch: FracDashpotMaxwell
    dashpot_c: Dashpot
    diagram = """
                    ___                 
                 ____| |___________╱╲__________
                |   _|_|  ca       ╲╱  cb, b   |
            ____|                              |____
                |              ___             |
                |_______________| |____________|
                               _|_|  cc
            """

    def __init__(self, params: FracJeffreysZenerParams) -> None:
        super().__init__()
        self.params = params
        self.maxwell_branch = FracDashpotMaxwell(
            FracDashpotMaxwellParams(params.dashpot_a, params.springpot_b)
        )
        self.dashpot_a = self.maxwell_branch.dashpot
        self.springpot_b = self.maxwell_branch.springpot
        self.dashpot_c = Dashpot(params.dashpot_c)

    @property
    def params(self) -> FracJeffreysZenerParams:
        return self._params

    @params.setter
    def params(self, params: FracJeffreysZenerParams) -> None:
        if not isinstance(params, FracJeffreysZenerParams):
            raise ValueError(
                "Invalid parameters: Expected FracJeffreysZenerParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.dashpot_c.G(t) + self.maxwell_branch.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        if t == 0:
            return 0

        ca = self.params.dashpot_a.c
        b = self.params.springpot_b.e
        cb = self.params.springpot_b.ce
        cc = self.params.dashpot_c.c

        def J_hat(s):
            numerator = ca * s + cb * s ** b
            denominator = (ca * s) * (cb * s ** b) + \
                cc * s * (ca * s + cb * s ** b)
            return (1 / s) * numerator / denominator

        t_mp = mpmath.mpf(t)
        J_t = mpmath.invertlaplace(J_hat, t_mp, method='talbot')
        return float(J_t)


@dataclass
class FracSolidZenerParams():
    dashpot_a: DashpotParams
    springpot_b: SpringpotParams
    spring_c: SpringParams


class FracSolidZener(Model):
    _params: FracSolidZenerParams
    dashpot_a: Dashpot
    springpot_b: Springpot
    maxwell_branch: FracDashpotMaxwell
    spring_c: Spring
    diagram = """
                      ___                 
                 ______| |__________╱╲_________
                |     _|_|  ca      ╲╱  cb, b  |
            ____|                              |____
                |                              |
                |__________╱╲  ╱╲  ╱╲  ________|
                             ╲╱  ╲╱  ╲╱  kc
            """

    def __init__(self, params: FracSolidZenerParams) -> None:
        super().__init__()
        self.params = params
        self.maxwell_branch = FracDashpotMaxwell(
            FracDashpotMaxwellParams(params.dashpot_a, params.springpot_b)
        )
        self.dashpot_a = self.maxwell_branch.dashpot
        self.springpot_b = self.maxwell_branch.springpot
        self.spring_c = Spring(params.spring_c)

    @property
    def params(self) -> FracSolidZenerParams:
        return self._params

    @params.setter
    def params(self, params: FracSolidZenerParams) -> None:
        if not isinstance(params, FracSolidZenerParams):
            raise ValueError(
                "Invalid parameters: Expected FracSolidZenerParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.spring_c.G(t) + self.maxwell_branch.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        if t == 0:
            return 0

        ca = self.params.dashpot_a.c
        b = self.params.springpot_b.e
        cb = self.params.springpot_b.ce
        kc = self.params.spring_c.k

        def J_hat(s):
            A = ca / cb
            B = kc * A
            numerator = 1 + A * s ** (1 - b)
            denominator = ca + kc / s + B * s ** (-b)
            return (1 / s ** 2) * numerator / denominator

        t_mp = mpmath.mpf(t)
        J_t = mpmath.invertlaplace(J_hat, t_mp, method='talbot')
        return float(J_t)


@dataclass
class FracSLSZenerParams():
    springpot_a: SpringpotParams
    spring_b: SpringParams
    spring_c: SpringParams


class FracSLSZener(Model):
    _params: FracSLSZenerParams
    springpot_a: Springpot
    spring_b: Spring
    maxwell_branch: FracSpringMaxwell
    spring_c: Spring
    diagram = """                
                 ___╱╲________╱╲  ╱╲  ╱╲  _____
                |   ╲╱  ca, a   ╲╱  ╲╱  ╲╱  kb |
            ____|                              |____
                |                              |
                |__________╱╲  ╱╲  ╱╲  ________|
                             ╲╱  ╲╱  ╲╱  kc
            """

    def __init__(self, params: FracSLSZenerParams) -> None:
        super().__init__()
        self.params = params
        self.maxwell_branch = FracSpringMaxwell(
            FracSpringMaxwellParams(params.springpot_a, params.spring_b)
        )
        self.springpot_a = self.maxwell_branch.springpot
        self.spring_b = self.maxwell_branch.spring
        self.spring_c = Spring(params.spring_c)

    @property
    def params(self) -> FracSLSZenerParams:
        return self._params

    @params.setter
    def params(self, params: FracSLSZenerParams) -> None:
        if not isinstance(params, FracSLSZenerParams):
            raise ValueError(
                "Invalid parameters: Expected FracSLSZenerParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.spring_c.G(t) + self.maxwell_branch.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        if t == 0:
            return 0

        a = self.params.springpot_a.e
        ca = self.params.springpot_a.ce
        kb = self.params.spring_b.k
        kc = self.params.spring_c.k

        def J_hat(s):
            numerator = ca * s ** a + kb
            denominator = ca * kb * s ** a + kc * (ca * s ** a + kb)
            return (1 / s) * numerator / denominator

        t_mp = mpmath.mpf(t)
        J_t = mpmath.invertlaplace(J_hat, t_mp, method='talbot')
        return float(J_t)


@dataclass
class FracZenerParams():
    springpot_a: SpringpotParams
    springpot_b: SpringpotParams
    springpot_c: SpringpotParams


class FracZener(Model):
    _params: FracZenerParams
    springpot_a: Springpot
    springpot_b: Springpot
    maxwell_branch: FracMaxwell
    springpot_c: Springpot
    diagram = """               
                 ______╱╲___________╱╲_________
                |      ╲╱  ca, a    ╲╱  cb, b  |
            ____|                              |____
                |                              |
                |______________╱╲______________|
                               ╲╱  cc, c
            """

    def __init__(self, params: FracZenerParams) -> None:
        super().__init__()
        self.params = params
        self.maxwell_branch = FracMaxwell(
            FracMaxwellParams(params.springpot_a, params.springpot_b)
        )
        self.springpot_a = self.maxwell_branch.springpot_a
        self.springpot_b = self.maxwell_branch.springpot_b
        self.springpot_c = Springpot(params.springpot_c)

    @property
    def params(self) -> FracZenerParams:
        return self._params

    @params.setter
    def params(self, params: FracZenerParams) -> None:
        if not isinstance(params, FracZenerParams):
            raise ValueError("Invalid parameters: Expected FracZenerParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.springpot_c.G(t) + self.maxwell_branch.G(t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        if t == 0:
            return 0

        a = self.params.springpot_a.e
        ca = self.params.springpot_a.ce
        b = self.params.springpot_a.e
        cb = self.params.springpot_b.ce
        c = self.params.springpot_c.e
        cc = self.params.springpot_c.ce

        def J_hat(s):
            numerator = ca * s ** a + cb * s ** b
            denominator = (ca * s ** a) * (cb * s ** b) + \
                (cc * s ** c) * (ca * s ** a + cb * s ** b)
            return (1 / s) * numerator / denominator

        t_mp = mpmath.mpf(t)
        J_t = mpmath.invertlaplace(J_hat, t_mp, method='talbot')
        return float(J_t)
