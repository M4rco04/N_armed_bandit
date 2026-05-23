import numpy.typing as npt

import random
import numpy as np

from bandit.bandit import Bandit
from algorithms.algorithm import Algorithm


class EpsilonGreedy(Algorithm):
    epsilon: float
    Q: npt.NDArray[np.float64]
    N: npt.NDArray[np.float64]

    def __init__(self, bandit: Bandit, games: int = 300, epsilon: float = 0.1):
        super().__init__(bandit, games)
        self.Q = np.zeros(shape=self.arms, dtype=np.float64)
        self.N = np.zeros(shape=self.arms, dtype=np.float64)
        self.epsilon = epsilon

    def play(self) -> float:

        for i in range(self.games):
            arm = self.choose_arm(i)
            reward = self.bandit.pull(arm)
            self.total_reward += reward

            self.N[arm] += 1
            self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

            if i != 0:
                self.regret[i] = self.regret[i - 1]
            self.regret[i] += self.bandit.best - reward

        return self.total_reward

    def choose_arm(self, i) -> int:
        if i < self.arms:
            arm = i
        elif random.random() <= self.epsilon:
            arm = random.choice([i for i in range(self.arms)])
        else:
            arm = max([i for i in range(self.arms)], key=lambda x: self.Q[x])

        return arm
