from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

from bandit.bandit import Bandit


class Algorithm(ABC):
    bandit: Bandit
    games: int
    arms: int
    regret: npt.NDArray[np.float64]
    total_reward: float

    def __init__(self, bandit: Bandit, games: int = 300):
        self.bandit = bandit
        self.arms = bandit.n
        self.games = games
        self.regret = np.zeros(shape=games, dtype=np.float64)
        self.total_reward = 0

    @abstractmethod
    def play(self):
        pass
