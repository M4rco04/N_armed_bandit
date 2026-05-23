import numpy.typing as npt

import numpy as np
import random


class Bandit:
    n: int
    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]
    best: float

    def __init__(
        self, arms: int, mean: npt.NDArray[np.float64], std: npt.NDArray[np.float64]
    ):
        self.n = arms
        self.mean = mean.copy()
        self.std = std.copy()
        self.best = max(self.mean)

    def pull(self, arm: int) -> float:
        avg, std = self.mean[arm], self.std[arm]
        reward = random.gauss(avg, std)

        return reward
