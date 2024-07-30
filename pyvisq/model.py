import numpy as np
import numpy.typing as npt


class Model:
    I: float
    D: float
    L: float

    time: npt.NDArray[np.float64]
    strain: npt.NDArray[np.float64]
    stress: npt.NDArray[np.float64]

    def __init__(self):
        pass

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(
                f"The attribute '{key}' does not exist in this model.")

    def _get_time(self, D: float, L: float, dwell_size: int) -> npt.NDArray[np.float64]:
        time = np.linspace(0, D, 1001, endpoint=False)
        time = np.hstack((time, D + np.linspace(0, L, dwell_size)))
        return time

    def set_test(self, config: dict[str, float]) -> None:
        if all(k in config for k in ['I', 'D', 'L']):
            self.I = config['I']
            self.D = config['D']
            self.L = config['L']
        else:
            raise ValueError("Provide I, D, and L for test config.")

    def make(self) -> None:
        pass

    def run(self, dwell_size: int = 1001) -> None:
        pass

    def trigger_force(self, trigger: float, dwell_size: int = 10001) -> None:
        while True:
            self.run(dwell_size=dwell_size)
            approach = self.get_approach()
            stress = approach["stress"]
            max_stress = stress[-1]
            if max_stress < 0:
                self.stress = -1 * self.time
                break
            if np.isclose(max_stress, trigger, rtol=1e-3):
                break
            if max_stress > trigger:
                rate = self.I / self.D
                trigger_idx = np.argmin(np.abs(stress - trigger))
                trigger_idx = max(trigger_idx, 1)
                self.D = self.time[trigger_idx]
                self.I = self.D * rate
            elif max_stress < trigger:
                scale = trigger / max_stress
                self.I *= scale
                self.D *= scale

    def get_approach(self) -> dict[str, npt.NDArray[np.float64]]:
        idx = np.argmin(np.abs(self.time - self.D))
        approach = {
            'time': self.time[:idx+1],
            'strain': self.strain[:idx+1],
            'stress': self.stress[:idx+1]
        }
        return approach

    def get_dwell(self) -> dict[str, npt.NDArray[np.float64]]:
        idx = np.argmin(np.abs(self.time - self.D))
        dwell = {
            'time': self.time[idx:],
            'strain': self.strain[idx:],
            'stress': self.stress[idx:]
        }
        return dwell
