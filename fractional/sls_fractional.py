import copy
import numpy as np
import numpy.typing as npt
from scipy.special import gamma


def ml(a: float, b: float, z: npt.NDArray[np.float64], n_term: int = 50) -> npt.NDArray[np.float64]:
    z = np.asarray(z)
    result = np.zeros_like(z)
    for n in range(n_term):
        term = (z ** n) / gamma(a * n + b)
        result += term
    return result


class FractionalSLS:
    A: float
    B: float
    EK: float
    CA: float
    CB: float

    E: float
    T: float

    I: float
    D: float
    L: float

    time: npt.NDArray[np.float64]
    strain: npt.NDArray[np.float64]
    modulus: npt.NDArray[np.float64]
    stress: npt.NDArray[np.float64]

    def __init__(self, config: dict[str, float]) -> None:
        if 'A' not in config or 'B' not in config:
            raise ValueError("Missing configuration for A and/or B")

        self.A = max(config['A'], config['B'])
        self.B = min(config['A'], config['B'])

        if not (0 <= self.B < self.A <= 1):
            raise ValueError("Invalid values: Ensure 0 <= B < A <= 1")

        if 'CA' in config and 'CB' in config:
            self.CA = config['CA']
            self.CB = config['CB']
        elif all(k in config for k in ['E1', 'E2', 'T1', 'T2']):
            self.E1 = config['E1']
            self.E2 = config['E2']
            self.T1 = config['T1']
            self.T2 = config['T2']
        else:
            raise ValueError(
                "Either provide CA and CB or E1, E2, T1, and T2 for computation")

        if 'EK' not in config:
            raise ValueError("Missing configuration for EK")

        self.EK = config['EK']

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(
                f"The attribute '{key}' does not exist in this model.")

    def set_test(self, config: dict[str, float]) -> None:
        if all(k in config for k in ['I', 'D', 'L']):
            self.I = config['I']
            self.D = config['D']
            self.L = config['L']
        else:
            raise ValueError(
                "Provide I, D, and L for test config")

    def make(self) -> None:
        if all(hasattr(self, attr) for attr in ['E1', 'E2', 'T1', 'T2']):
            self.CA = self.E1 * self.T1 ** self.A
            self.CB = self.E2 * self.T2 ** self.B

        self.T = (self.CA / self.CB) ** (1 / (self.A - self.B))
        self.E = self.CA / (self.T ** self.A)

    def _get_time(self, D: float, L: float) -> npt.NDArray[np.float64]:
        time = np.linspace(0, 2 * D, 2001)
        time = np.hstack((time, 2 * D + np.linspace(0, L, 1001)))
        return time

    def run(self) -> None:
        self.time = self._get_time(self.D, self.L)
        u = np.heaviside
        self.strain = (self.I * self.time / self.D) * (1 - u(self.time - self.D, 0.5)) + \
            self.I * u(self.time - self.D, 0.5)

        self.modulus = np.zeros_like(self.time)
        self.modulus[1:] = self.E * (self.time[1:] / self.T) ** (-self.B) * \
            ml(self.A-self.B, 1-self.B, -(self.time[1:] / self.T)
               ** (self.A - self.B)) + self.EK
        self.stress = self.modulus * self.strain

    def get_approach(self) -> dict[str, npt.NDArray[np.float64]]:
        idx = np.argmin(np.abs(self.time - self.D))
        approach = {
            'time': self.time[:idx+1],
            'strain': self.strain[:idx+1],
            'modulus': self.modulus[:idx+1],
            'stress': self.stress[:idx+1]
        }
        return approach

    def get_dwell(self) -> dict[str, npt.NDArray[np.float64]]:
        idx = np.argmin(np.abs(self.time - self.D))
        dwell = {
            'time': self.time[idx:],
            'strain': self.strain[idx:],
            'modulus': self.modulus[idx:],
            'stress': self.stress[idx:]
        }
        return dwell


def sweep(model: FractionalSLS, x_to_y: tuple[str, str], values: list[float]) -> dict[float, FractionalSLS]:
    sweep_dict = dict()
    x, y = x_to_y
    y_value = getattr(model, y)
    for val in values:
        new_model = copy.deepcopy(model)
        x_value = val * y_value
        setattr(new_model, x, x_value)
        new_model.make()
        new_model.run()
        sweep_dict[val] = new_model

    return sweep_dict
