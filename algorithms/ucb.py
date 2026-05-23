import numpy.typing as npt

import numpy as np

from bandit.bandit import Bandit
from algorithms.algorithm import Algorithm


class UCB(Algorithm):
    mean: npt.NDArray[np.float64]
    N: npt.NDArray[np.float64]

    def __init__(self, bandit: Bandit, games: int = 300):
        super().__init__(bandit, games)
        self.mean = np.zeros(shape=self.arms, dtype=np.float64)
        self.N = np.zeros(shape=self.arms, dtype=np.float64)

    def play(self):

        for i in range(self.games):
            arm = max(range(self.arms), key=lambda x: self.ucb(x, i))

            reward = self.bandit.pull(arm)

            self.mean[arm] = (self.mean[arm] * self.N[arm] + reward) / (self.N[arm] + 1)
            self.N[arm] += 1

            self.total_reward += reward
            if i != 0:
                self.regret[i] = self.regret[i - 1]
            self.regret[i] += self.bandit.best - reward

        return self.total_reward

    def ucb(self, id: int, n: int) -> float:
        if self.N[id] == 0:
            return float("inf")
        return self.mean[id] + self.g(n) / np.sqrt(self.N[id])

    def g(self, n: int) -> float:
        return np.sqrt(2 * np.log(n))
