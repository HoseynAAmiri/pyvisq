import numpy as np

from .model import Model


class SLS(Model):
    E1: float
    T1: float
    EK: float

    def __init__(self, config: dict[str, float]) -> None:
        super().__init__()
        required_attrs = [attr for attr in self.__annotations__]

        for attr in required_attrs:
            if attr in config:
                setattr(self, attr, config[attr])
            else:
                raise ValueError(
                    f"Missing required configuration for '{attr}'")

    def run(self, dwell_size: int = 1001) -> None:
        self.time = self._get_time(self.D, self.L, dwell_size)
        u = np.heaviside
        self.strain = (self.I * self.time / self.D) * (1 - u(self.time - self.D, 0.5)) + \
            self.I * u(self.time - self.D, 0.5)
        self.modulus = np.zeros_like(self.time)

        self.stress = self.I*(-self.E1*self.T1*(-np.exp(self.D/self.T1) + np.exp(self.time/self.T1))*u(-self.D + self.time, 0.5) + self.E1*self.time*np.exp(self.time/self.T1) - self.E1*(-self.D + self.T1*np.exp((self.D - self.time)/self.T1) - self.T1 + self.time)*np.exp(self.time/self.T1)
                              * u(-self.D + self.time, 0.5) - self.EK*self.T1*(-np.exp(self.D/self.T1) + np.exp(self.time/self.T1))*u(-self.D + self.time, 0.5) + self.EK*self.T1*np.exp(self.time/self.T1) - self.EK*self.T1)*np.exp(-self.time/self.T1)/self.D
