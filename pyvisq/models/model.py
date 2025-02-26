import numpy as np
import numpy.typing as npt

from typing import Callable
from functools import lru_cache
from scipy.integrate import quad
from enum import Enum
from dataclasses import dataclass, field


class TestMethod(Enum):
    CREEP = 'creep'
    RELAXATION = 'relaxation'


@dataclass
class Test:
    """
    ```
    input profile:
        I   ==========
          //:        :\\
         // :        : \\
        //  :D1      :L1\\D2==========L2
    ```
    """
    method: TestMethod
    I: float
    D1: float
    L1: float
    D2: float
    L2: float

    def __post_init__(self):
        if self.D1 <= 0:
            raise ValueError("D1 must be greater than zero.")
        if self.L1 <= 0:
            raise ValueError("L1 must be greater than zero.")


@dataclass
class Data:
    time: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )
    strain: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )
    stress: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )


class Model:
    params: dict[str, float] = {}
    diagram = ""
    _test: Test
    data: Data

    def __init__(self):
        self.data = Data()

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(
                f"The attribute '{key}' does not exist in this model."
            )

    def __str__(self) -> str:
        return self.diagram + "\n" + str(self.params)

    def G(self, t: float) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement relaxation modulus."
        )

    def J(self, t: float) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement creep compliance."
        )

    def set_time(
        self,
        D1: float | None = None,
        L1: float | None = None,
        D2: float | None = None,
        L2: float | None = None,
        D1_size: int = 21,
        L1_size: int = 21,
        D2_size: int = 21,
        L2_size: int = 21,
    ) -> None:
        D1 = self.test.D1 if D1 is None else D1
        L1 = self.test.L1 if L1 is None else L1
        D2 = self.test.D2 if D2 is None else D2
        L2 = self.test.L2 if L2 is None else L2
        t1 = np.linspace(0, D1, D1_size, endpoint=True)
        t2 = D1 + np.linspace(0, L1, L1_size, endpoint=True)
        if D2 == 0:
            t3 = np.array([])
            t4 = np.array([])
        else:
            t3 = D1 + L1 + np.linspace(0, D2, D2_size, endpoint=True)
            if L2 != 0:
                t4 = D1 + L1 + D2 + np.linspace(0, L2, L2_size, endpoint=True)
        self.data.time = np.unique(
            np.concatenate((t1, t2, t3, t4), dtype=float)
        )

    def set_test(self, test: Test) -> None:
        self.test = test

    @property
    def test(self) -> Test:
        return self._test

    @test.setter
    def test(self, test: Test) -> None:
        if not isinstance(test, Test):
            raise ValueError("test is not a Test object.")

        self._test = test

    def time_between(self, first: float, last: float) -> npt.NDArray[np.float64]:
        time = self.data.time
        return time[(first <= time) & (time <= last)]

    def _input(self) -> npt.NDArray[np.float64]:
        input_D1 = self.time_between(
            0, self.test.D1) * self.test.I / self.test.D1
        input_L1 = self.test.I * \
            np.ones_like(self.time_between(
                self.test.D1, self.test.L1 + self.test.D1))
        if self.test.D2 == 0:
            input_D2 = np.array([])
            input_L2 = np.array([])
        else:
            first = self.test.L1 + self.test.D1
            last = self.test.L1 + self.test.D1 + self.test.D2
            input_D2 = self.test.I * \
                (1 - (self.time_between(first, last) -
                 self.test.L1 - self.test.D1) / self.test.D2)
            if self.test.L2 != 0:
                first = self.test.L1 + self.test.D1 + self.test.D2
                last = self.test.L1 + self.test.D1 + self.test.D2 + self.test.L2
                input_L2 = np.zeros_like(self.time_between(first, last))
        return np.concatenate((input_D1, input_L1[1:], input_D2[1:], input_L2[1:]), dtype=float)

    @lru_cache(maxsize=100)
    def _input_rate(self, t: float) -> float:
        if t < self.test.D1:
            return self.test.I / self.test.D1
        if self.test.L1 + self.test.D1 <= t < self.test.D1 + self.test.L1 + self.test.D2:
            return - self.test.I / self.test.D2
        return 0.0

    def set_input(self) -> None:
        if self.test.method == TestMethod.CREEP:
            self.data.stress = self._input()
        if self.test.method == TestMethod.RELAXATION:
            self.data.strain = self._input()

    @staticmethod
    @lru_cache(maxsize=100)
    def quad(func: Callable, a: float, b: float) -> float:
        return quad(func, a, b, epsrel=1e-3)[0]

    def get_output(self, t: float) -> float:
        if self.test.method == TestMethod.CREEP:
            func = self.J
        if self.test.method == TestMethod.RELAXATION:
            func = self.G

        def _integrand(t_prime: float) -> float:
            return func(t - t_prime) * self._input_rate(t_prime)

        def _boltzmann_integral(t: float) -> float:
            t = round(t, max(0, int(-np.log10(self.test.D1))) + 4)
            if t < 0:
                raise ValueError("Invalid time value.")
            if t == 0:
                return 0.0
            if t < self.test.D1:
                return Model.quad(_integrand, 0, t)
            else:
                p1 = Model.quad(_integrand, 0, self.test.D1)
                if t < self.test.D1 + self.test.L1:
                    return p1
                else:
                    if t < self.test.D1 + self.test.L1 + self.test.D2:
                        return p1 + Model.quad(_integrand, self.test.L1 + self.test.D1, t)
                    else:
                        p2 = Model.quad(
                            _integrand,
                            self.test.L1 + self.test.D1,
                            self.test.D1 + self.test.L1 + self.test.D2,
                        )
                        return p1 + p2
        return _boltzmann_integral(t)

    def run(self) -> None:
        if len(self.data.time) == 0:
            raise ValueError("Set model time before the run.")
        if self.test.method == TestMethod.CREEP:
            self.data.strain = np.vectorize(self.get_output)(self.data.time)
        if self.test.method == TestMethod.RELAXATION:
            self.data.stress = np.vectorize(self.get_output)(self.data.time)

    def get_data(self, first: float, last: float) -> Data:
        idx1 = np.argmin(np.abs(self.data.time - first))
        idx2 = np.argmin(np.abs(self.data.time - last))
        return Data(
            time=self.data.time[idx1:idx2+1],
            strain=self.data.strain[idx1:idx2+1],
            stress=self.data.stress[idx1:idx2+1]
        )

    # def trigger_force(self, trigger: float) -> None:
    #     while True:
    #         self.run()
    #         approach = self.get_approach()
    #         stress = approach["stress"]
    #         max_stress = stress[-1]
    #         if max_stress < 0:
    #             self.stress = -1 * self.time
    #             break
    #         if np.isclose(max_stress, trigger, rtol=1e-3):
    #             break
    #         if max_stress > trigger:
    #             rate = self.I / self.D
    #             trigger_idx = np.argmin(np.abs(stress - trigger)).astype(int)
    #             trigger_idx = int(max(trigger_idx, 1))
    #             self.D = self.time[trigger_idx]
    #             self.I = self.D * rate
    #         elif max_stress < trigger:
    #             scale = trigger / max_stress
    #             self.I *= scale
    #             self.D *= scale
