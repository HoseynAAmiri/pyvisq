import mpmath

from functools import lru_cache
from dataclasses import dataclass

from .model import Model
from .elements import Spring, SpringParams
from .maxwell import Maxwell, MaxwellParams


@dataclass
class GeneralizedParams():
    spring1: SpringParams
    branches: list[MaxwellParams]


class Generalized(Model):
    _params: GeneralizedParams
    spring1: Spring
    branches: list[Maxwell]
    diagram = """                
                 __________╱╲  ╱╲  ╱╲  ________
                |            ╲╱  ╲╱  ╲╱  k1    |
                |                              |
                |                     ___      |
            ____|__╱╲  ╱╲  ╱╲  ________| |_____|____
                |    ╲╱  ╲╱  ╲╱  k2   _|_|  c2 |
                :                :          :  :
                :                :    ___  :   : 
                |__╱╲  ╱╲  ╱╲  ________| |_____|
                     ╲╱  ╲╱  ╲╱  kn   _|_|  cn  
            """

    def __init__(self, params: GeneralizedParams) -> None:
        super().__init__()
        self.params = params
        self.spring1 = Spring(params.spring1)
        self.branches = [Maxwell(branch) for branch in params.branches]

    @property
    def params(self) -> GeneralizedParams:
        return self._params

    @params.setter
    def params(self, params: GeneralizedParams) -> None:
        if not isinstance(params, GeneralizedParams):
            raise ValueError("Invalid parameters: Expected GeneralizedParams.")
        self._params = params

    @lru_cache(maxsize=100)
    def G(self, t: float) -> float:
        return self.spring1.G(t) + sum([branch.G(t) for branch in self.branches])

    @lru_cache(maxsize=100)
    def J(self, t: float) -> float:
        if t == 0:
            return 0

        if len(self.branches) == 0:
            return self.spring1.J(t)

        def J_hat(s):
            all_branches = self.spring1.params.k / s
            for branch in self.branches:
                branch_params = branch.params
                k = branch_params.spring.k
                c = branch_params.dashpot.c
                tau = c / k
                all_branches += k / (s + 1 / tau)
            return (1 / s ** 2) / all_branches

        t_mp = mpmath.mpf(t)
        J_t = mpmath.invertlaplace(J_hat, t_mp, method='talbot')
        return float(J_t)
