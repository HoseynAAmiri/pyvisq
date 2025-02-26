# this is python version of poynting_thomson.jl in the folder "pyvisq/models"
import mpmath

import numpy as np

from functools import lru_cache
from dataclasses import dataclass

from .model import Model
from .elements import Spring, Dashpot, Springpot, SpringParams, DashpotParams, SpringpotParams
from .kelvinvoigt import KelvinVoigt, KelvinVoigtParams, FracDashpotKelvinVoigt, FracDashpotKelvinVoigtParams, FracSpringKelvinVoigt, FracSpringKelvinVoigtParams, FracKelvinVoigt, FracKelvinVoigtParams
from ..utils import MittagLeffler


@dataclass
class SLSPTParams():
    dashpot_a: DashpotParams
    spring_b: SpringParams
    spring_c: SpringParams


class SLSPT(Model):
    _params: SLSPTParams
    dashpot_a: Dashpot
    spring_b: Spring
    kelvingvoigt_branch: KelvinVoigt
    spring_c: Spring
    diagram = """
                         ___
                 _________| |________
                |        _|_|   ca   |
            ____|                    |______╱╲  ╱╲  ╱╲  ____
                |                    |        ╲╱  ╲╱  ╲╱  kc
                |____╱╲  ╱╲  ╱╲  ____|
                       ╲╱  ╲╱  ╲╱  kb
            """

    def __init__(self, params: SLSPTParams) -> None:
        super().__init__()
        self.params = params
        self.kelvingvoigt_branch = KelvinVoigt(
            KelvinVoigtParams(params.dashpot_a, params.spring_b)
        )
        self.dashpot_a = self.kelvingvoigt_branch.dashpot
        self.spring_b = self.kelvingvoigt_branch.spring
        self.spring_c = Spring(params.spring_c)

    @property
    def params(self) -> SLSPTParams:
        return self._params

    @params.setter
    def params(self, params: SLSPTParams) -> None:
        if not isinstance(params, SLSPTParams):
            raise ValueError("Invalid parameters: Expected SLSPTParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        ca = self.params.dashpot_a.c
        kb = self.params.spring_b.k
        kc = self.params.spring_c.k

        Zkb = (kc**2) / (kb + kc)
        Zca = ca * (kc**2) / (kb + kc)**2
        Zkc = (kb * kc) / (kb + kc)
        return Zkc + Zkb * np.exp(-t * Zkb / Zca)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.kelvingvoigt_branch.J(t) + self.spring_c.J(t)


@dataclass
class JeffreysPTParams():
    dashpot_a: DashpotParams
    spring_b: SpringParams
    dashpot_c: DashpotParams


class JeffreysPT(Model):
    _params: JeffreysPTParams
    dashpot_a: Dashpot
    spring_b: Spring
    kelvingvoigt_branch: KelvinVoigt
    dashpot_c: Dashpot
    diagram = """
                         ___
                 _________| |________
                |        _|_|  ca    |      __
            ____|                    |_______| |____
                |                    |      _|_|  cc
                |____╱╲  ╱╲  ╱╲  ____|
                       ╲╱  ╲╱  ╲╱  kb
            """

    def __init__(self, params: JeffreysPTParams) -> None:
        super().__init__()
        self.params = params
        self.kelvingvoigt_branch = KelvinVoigt(
            KelvinVoigtParams(params.dashpot_a, params.spring_b)
        )
        self.dashpot_a = self.kelvingvoigt_branch.dashpot
        self.spring_b = self.kelvingvoigt_branch.spring
        self.dashpot_c = Dashpot(params.dashpot_c)

    @property
    def params(self) -> JeffreysPTParams:
        return self._params

    @params.setter
    def params(self, params: JeffreysPTParams) -> None:
        if not isinstance(params, JeffreysPTParams):
            raise ValueError("Invalid parameters: Expected JeffreysPTParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        ca = self.params.dashpot_a.c
        kb = self.params.spring_b.k
        cc = self.params.dashpot_c.c

        Zca = (cc**2) / (ca + cc)
        Zkb = kb * (cc**2) / (ca + cc)**2
        Zcc = (ca * cc) / (ca + cc)
        diracterm = 0.0 if t != 0.0 else np.inf
        return Zkb * np.exp(-Zkb * t / Zca) + Zcc * diracterm

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.kelvingvoigt_branch.J(t) + self.dashpot_c.J(t)


@dataclass
class FracSLSPTParams():
    springpot_a: SpringpotParams
    spring_b: SpringParams
    spring_c: SpringParams


class FracSLSPT(Model):
    _params: FracSLSPTParams
    springpot_a: Springpot
    spring_b: Spring
    kelvingvoigt_branch: FracSpringKelvinVoigt
    spring_c: Spring
    ml: MittagLeffler
    diagram = """
                 _________╱╲_________
                |         ╲╱  ca, a  |
            ____|                    |______╱╲  ╱╲  ╱╲  ____
                |                    |        ╲╱  ╲╱  ╲╱  kc
                |____╱╲  ╱╲  ╱╲  ____|
                       ╲╱  ╲╱  ╲╱  kb
            """

    def __init__(self, params: FracSLSPTParams) -> None:
        super().__init__()
        self.params = params
        self.kelvingvoigt_branch = FracSpringKelvinVoigt(
            FracSpringKelvinVoigtParams(params.springpot_a, params.spring_b)
        )
        self.springpot_a = self.kelvingvoigt_branch.springpot
        self.spring_b = self.kelvingvoigt_branch.spring
        self.spring_c = Spring(params.spring_c)
        self.ml = MittagLeffler()

    @property
    def params(self) -> FracSLSPTParams:
        return self._params

    @params.setter
    def params(self, params: FracSLSPTParams) -> None:
        if not isinstance(params, FracSLSPTParams):
            raise ValueError("Invalid parameters: Expected FracSLSPTParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        t = max(t, 1e-10)
        a = self.params.springpot_a.e
        ca = self.params.springpot_a.ce
        kb = self.params.spring_b.k
        kc = self.params.spring_c.k

        Zkb = (kc**2) / (kb + kc)
        Zca = ca * (kc**2) / (kb + kc)**2
        Zkc = (kb * kc) / (kb + kc)
        return Zkb * self.ml(a, a, - (Zkb / Zca) * (t**a)) + Zkc

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.kelvingvoigt_branch.J(t) + self.spring_c.J(t)


@dataclass
class FracJeffreysPTParams():
    dashpot_a: DashpotParams
    springpot_b: SpringpotParams
    dashpot_c: DashpotParams


class FracJeffreysPT(Model):
    _params: FracJeffreysPTParams
    dashpot_a: Dashpot
    springpot_a: Springpot
    kelvingvoigt_branch: FracDashpotKelvinVoigt
    dashpot_c: Dashpot
    ml: MittagLeffler
    diagram = """
                         ___
                 _________| |________
                |        _|_|  ca    |        ___
            ____|                    |_________| |_____
                |                    |        _|_|  cc
                |_________╱╲_________|
                          ╲╱  cb, b
            """

    def __init__(self, params: FracJeffreysPTParams) -> None:
        super().__init__()
        self.params = params
        self.kelvingvoigt_branch = FracDashpotKelvinVoigt(
            FracDashpotKelvinVoigtParams(params.dashpot_a, params.springpot_b)
        )
        self.dashpot_a = self.kelvingvoigt_branch.dashpot
        self.springpot_a = self.kelvingvoigt_branch.springpot
        self.dashpot_c = Dashpot(params.dashpot_c)
        self.ml = MittagLeffler()

    @property
    def params(self) -> FracJeffreysPTParams:
        return self._params

    @params.setter
    def params(self, params: FracJeffreysPTParams) -> None:
        if not isinstance(params, FracJeffreysPTParams):
            raise ValueError(
                "Invalid parameters: Expected FracJeffreysPTParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        t = max(t, 1e-10)
        ca = self.params.dashpot_a.c
        b = self.params.springpot_b.e
        cb = self.params.springpot_b.ce
        cc = self.params.dashpot_c.c

        Zca = (cc**2) / (ca + cc)
        Zcb = cb * (cc**2) / (ca + cc)**2
        Zcc = (ca * cc) / (ca + cc)
        diracterm = 0.0 if t != 0.0 else np.inf
        return Zcb * (t**(-b)) * self.ml(1 - b, 1 - b, -Zcb * t**(1 - b) / Zca) + Zcc * diracterm

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.kelvingvoigt_branch.J(t) + self.dashpot_c.J(t)


@dataclass
class FracPTParams():
    springpot_a: SpringpotParams
    springpot_b: SpringpotParams
    springpot_c: SpringpotParams


class FracPT(Model):
    _params: FracPTParams
    springpot_a: Springpot
    springpot_b: Springpot
    kelvingvoigt_branch: FracKelvinVoigt
    springpot_c: Springpot
    diagram = """
                 _________╱╲_________
                |         ╲╱  ca, a  |
            ____|                    |______╱╲______
                |                    |      ╲╱ cc, c
                |_________╱╲_________|
                          ╲╱  cb, b
            """

    def __init__(self, params: FracPTParams) -> None:
        super().__init__()
        self.params = params
        self.kelvingvoigt_branch = FracKelvinVoigt(
            FracKelvinVoigtParams(params.springpot_a, params.springpot_b)
        )
        self.springpot_a = self.kelvingvoigt_branch.springpot_a
        self.springpot_b = self.kelvingvoigt_branch.springpot_b
        self.springpot_c = Springpot(params.springpot_c)

    @property
    def params(self) -> FracPTParams:
        return self._params

    @params.setter
    def params(self, params: FracPTParams) -> None:
        if not isinstance(params, FracPTParams):
            raise ValueError("Invalid parameters: Expected FracPTParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        if t == 0:
            return 0

        a = self.params.springpot_a.e
        ca = self.params.springpot_a.ce
        b = self.params.springpot_b.e
        cb = self.params.springpot_b.ce
        c = self.params.springpot_c.e
        cc = self.params.springpot_c.ce

        def G_hat(s):
            numerator = cc * s**c * (ca * s**a + cb * s**b)
            denominator = cc * s**c + ca * s**a + cb * s**b
            return (1 / s) * numerator / denominator

        t_mp = mpmath.mpf(t)
        G_t = mpmath.invertlaplace(G_hat, t_mp, method='talbot')
        return float(G_t)

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        return self.kelvingvoigt_branch.J(t) + self.springpot_c.J(t)
